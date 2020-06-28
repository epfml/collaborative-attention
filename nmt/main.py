"""Training Transformer for WMT17 Dataset

This implements the machine translation benchmark tasks,
# TODO add link to docs
"""
import argparse
import json
import logging
import os
import time
from copy import deepcopy
import unittest.mock as mock

import numpy as np
import torch
import torch.distributed as dist

# Parser is defined and called before the rest of the import
# because we need to inject the collaborative heads into mlbench_core
# with `use_collaborative_heads` and `key_query_dim` parameter
# before importing the rest of the modules
parser = argparse.ArgumentParser(description="Process run parameters")
parser.add_argument("--run_id", type=str, default="1", help="The id of the run")
parser.add_argument(
    "--root-dataset",
    type=str,
    default="/datasets",
    help="Default root directory to dataset.",
)
parser.add_argument(
    "--root-checkpoint",
    type=str,
    default="/checkpoint",
    help="Default root directory to checkpoint.",
)
parser.add_argument(
    "--root-output",
    type=str,
    default="/output",
    help="Default root directory to output.",
)
parser.add_argument(
    "--validation_only",
    action="store_true",
    default=False,
    help="Only validate from checkpoints.",
)
parser.add_argument("--gpu", action="store_true", default=False, help="Train with GPU")
parser.add_argument(
    "--light",
    action="store_true",
    default=False,
    help="Train to light target metric goal",
)
parser.add_argument("--rank", type=int, default=1, help="The rank of the process")
parser.add_argument(
    "--backend", type=str, default="mpi", help="PyTorch distributed backend"
)
parser.add_argument("--hosts", type=str, help="The list of hosts")
parser.add_argument("--uid", type=str, default="allreduce", help="Name of the run")

# Collaborative heads
parser.add_argument(
    "--use_collaborative_heads",
    action="store_true",
    default=False,
    help="Use collaborative heads instead of concatenation",
)

parser.add_argument(
    "--key_query_dim",
    type=int,
    default=None,
    help="Dimension of the key/query projections, default is the hidden size of the transformer",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Inject the collaborative head into mlbench-core transformer.
    # We specify key_query_dim and use_collaborative_heads with a partial application
    # to make the signature of CollaborativeMultiheadAttention exactly the same as MultiheadAttention
    if args.use_collaborative_heads or args.key_query_dim is not None:
        import mlbench_core.models.pytorch.transformer.modules.multihead_attention
        from collab_multihead_attention import CollaborativeMultiheadAttention
        import functools
        import sys

        mlbench_core.models.pytorch.transformer.modules.multihead_attention.MultiheadAttention = functools.partial(
            CollaborativeMultiheadAttention,
            key_query_dim=args.key_query_dim,
            use_collaborative_heads=args.use_collaborative_heads,
        )

        reload_mods = [
            mod
            for mod in sys.modules
            if "mlbench" in mod and "multihead_attention" not in mod
        ]
        for mod in reload_mods:
            del sys.modules[mod]

from mlbench_core.controlflow.pytorch.checkpoints_evaluation import (
    CheckpointsEvaluationControlFlow,
)
from mlbench_core.controlflow.pytorch.controlflow import (
    record_train_batch_stats,
    record_validation_stats,
)
from mlbench_core.dataset.nlp.pytorch import WMT17Dataset, get_batches
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.goals import task4_time_to_bleu_goal
from mlbench_core.evaluation.pytorch.criterion import LabelSmoothing
from mlbench_core.evaluation.pytorch.metrics import BLEUScore
from mlbench_core.lr_scheduler.pytorch.lr import SQRTTimeDecayLRWithWarmup
from mlbench_core.models.pytorch.transformer import SequenceGenerator, TransformerModel
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq
from torch.utils.data import DataLoader

from utils import (
    Arguments,
    build_optimizer,
    compute_loss,
    get_full_batch_size,
    opt_step,
    prepare_batch,
    validation_round,
)

try:
    import horovod.torch as hvd
except ImportError as e:
    hvd = None


def equalize_batches(batches, world_size, seed):
    """Given a list of batches, makes sure each workers has equal number
    by adding new batches using bootstrap sampling

    Args:
        batches (list): The list of batches
        world_size (int): Distributed world size
        seed (int): Random seed to use (must be the same across all workers)

    Returns:
        (list): The new extended batches
    """
    to_add = world_size - (len(batches) % world_size)
    if to_add == 0:
        return batches
    np.random.seed(seed)
    bootstrapped = np.random.choice(np.arange(len(batches)), size=to_add)

    to_add = [batches[i] for i in bootstrapped]
    return batches + to_add


def get_max_tokens(world_size, update_freq, max_tokens_batch=2 ** 17):
    """Returns the max number of tokens a batch should have

    Args:
        world_size (int): Distributed world size
        update_freq (int): Update frequency (min 1)
        max_tokens_batch (int): Max tokens per batch over all workers

    Returns:
        (int): The max number of tokens per batch per worker
    """
    return int(max_tokens_batch / (world_size * update_freq))


logger = logging.getLogger("mlbench")


DEFAULT_TRANSFORMER_ARCH = {
    "max_source_positions": 256,
    "max_target_positions": 256,
    "dropout": 0.3,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "encoder_embed_path": None,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_layers": 6,
    "encoder_attention_heads": 16,
    "encoder_normalize_before": True,
    "encoder_learned_pos": False,
    "decoder_embed_path": None,
    "decoder_embed_dim": 1024,
    "decoder_ffn_embed_dim": 4096,
    "decoder_layers": 6,
    "decoder_attention_heads": 16,
    "decoder_learned_pos": False,
    "decoder_normalize_before": True,
    "share_decoder_input_output_embed": False,
    "share_all_embeddings": False,
    "no_token_positional_embeddings": False,
    "softmax_type": None,
}


def train_loop(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    validation_only=False,
    use_cuda=False,
    light_target=False,
    seed=42,
):
    """Train loop"""
    train_epochs = 9

    math_mode = "fp16"
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Dataset arguments
    update_freq = max(16 // world_size, 1)
    max_tokens = get_max_tokens(world_size, update_freq)
    max_source_positions, max_target_positions = 80, 80
    seq_len_multiple = 2
    left_pad = (True, False)
    lang = ("en", "de")

    # specific arch
    model_args = deepcopy(DEFAULT_TRANSFORMER_ARCH)
    model_args["max_source_positions"] = max_source_positions
    model_args["max_target_positions"] = max_target_positions
    model_args["share_all_embeddings"] = True
    model_args["dropout"] = 0.1
    model_args["softmax_type"] = "fast_fill"

    lr = 1.976e-3
    optimizer_args = {"lr": lr, "eps": 1e-9, "betas": (0.9, 0.98)}
    scheduler_args = {"base_lr": lr, "warmup_init_lr": 0.0, "warmup_steps": 1000}

    loss_scaling_fp16 = {
        "init_scale": 2.0 ** 7,
        "scale_factor": 2,
        "scale_window": 2000,
    }

    criterion_args = {"smoothing": 0.1, "fast_xentropy": True}

    # Horovod stuff
    use_horovod = (math_mode == "fp16") and dist.get_backend() == dist.Backend.MPI
    if use_horovod:
        hvd.init()
        logger.info("Using horovod rank={}".format(hvd.rank()))
        tensor = torch.tensor([1])
        res = hvd.allreduce(tensor, op=hvd.Sum)
        assert res[0] == world_size

    # Load train and validation datasets
    train_set = WMT17Dataset(
        dataset_dir,
        download=True,
        train=True,
        shuffle=True,
        lang=lang,
        left_pad=left_pad,
        max_positions=(max_source_positions, max_target_positions),
        seq_len_multiple=seq_len_multiple,
    )

    validation_set = WMT17Dataset(
        dataset_dir,
        download=False,
        test=True,
        shuffle=True,
        lang=lang,
        left_pad=left_pad,
        max_positions=(max_source_positions, max_target_positions),
        seq_len_multiple=seq_len_multiple,
    )
    src_dict, trg_dict = train_set.src_dict, train_set.trg_dict

    train_batches = get_batches(
        train_set, max_tokens=max_tokens, bsz_mult=8, shuffle=True, seed=seed
    )
    val_batches = get_batches(
        validation_set, max_tokens=max_tokens, bsz_mult=8, shuffle=False
    )

    train_batches = equalize_batches(train_batches, world_size, seed=seed)

    # Partition by rank
    train_batches = partition_dataset_by_rank(train_batches, rank, world_size)
    val_batches = partition_dataset_by_rank(val_batches, rank, world_size)

    total_train_points = sum(len(b) for b in train_batches)

    validate_every = update_freq * round(
        len(train_batches) * 0.30 / update_freq
    )  # Validate every 30%

    assert (validate_every % update_freq) == 0
    logger.info(
        "Using {} total train points, {} batches".format(
            total_train_points, len(train_batches)
        )
    )

    train_loader = DataLoader(
        train_set,
        num_workers=1,
        pin_memory=False,
        collate_fn=train_set.collater,
        batch_sampler=train_batches,
    )

    val_loader = DataLoader(
        validation_set,
        num_workers=1,
        pin_memory=False,
        collate_fn=validation_set.collater,
        batch_sampler=val_batches,
    )

    model = TransformerModel(Arguments(model_args), src_dict, trg_dict)
    criterion = LabelSmoothing(padding_idx=src_dict.pad(), **criterion_args)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    fp_optimizer, optimizer, model = build_optimizer(
        model,
        optimizer_args,
        math_mode=math_mode,
        scaling_args=loss_scaling_fp16,
        use_horovod=use_horovod,
        use_cuda=use_cuda,
    )

    scheduler = SQRTTimeDecayLRWithWarmup(optimizer, **scheduler_args)

    metrics = [BLEUScore(use_raw=True)]
    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.BEST
    )

    translator = SequenceGenerator(
        model,
        src_dict=deepcopy(src_dict),
        trg_dict=deepcopy(trg_dict),
        beam_size=4,
        stop_early=True,
        normalize_scores=True,
        len_penalty=0.6,
        sampling=False,
        sampling_topk=-1,
        minlen=1,
    )
    if not validation_only:

        goal = None
        # if light_target:
        #     goal = task4_time_to_bleu_goal(20)
        # else:
        #     goal = task4_time_to_bleu_goal(25)

        num_batches_per_device_train = len(train_loader)
        tracker = Tracker(metrics, run_id, rank, goal=goal)

        dist.barrier()
        tracker.start()

        for epoch in range(0, train_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.train()
            tracker.train()

            iter_sample_size = 0
            for batch_idx, sample in enumerate(train_loader):
                sample = prepare_batch(sample, use_cuda=use_cuda)
                tracker.batch_start()

                is_last = batch_idx == len(train_loader)
                update = (batch_idx % update_freq) == update_freq - 1
                init = (batch_idx % update_freq) == 0
                # Clear gradients in the optimizer.
                if init:
                    fp_optimizer.zero_grad()
                    iter_sample_size = 0
                    tracker.record_batch_init()

                # Compute the output
                output = model(**sample["net_input"])
                tracker.record_batch_fwd_pass()

                loss, sample_size = compute_loss(sample, output, criterion)
                loss_per_sample = loss.item() / sample_size
                iter_sample_size += sample_size
                tracker.record_batch_comp_loss()

                # Backprop
                fp_optimizer.backward_loss(loss)
                tracker.record_batch_backprop()

                if update or is_last:
                    # Optimize
                    full_bs = get_full_batch_size(
                        iter_sample_size, world_size=world_size, use_cuda=use_cuda
                    )

                    updated = opt_step(
                        fp_optimizer, full_bs, update_freq, math_mode, world_size
                    )
                    tracker.record_batch_opt_step()

                    if updated:
                        scheduler.step()

                tracker.batch_end()

                record_train_batch_stats(
                    batch_idx=batch_idx,
                    loss=loss_per_sample,
                    output=torch.Tensor([0]),
                    target=None,
                    metrics=[],
                    tracker=tracker,
                    num_batches_per_device_train=num_batches_per_device_train,
                )

                if (batch_idx + 1) % validate_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    metric_values, loss = validation_round(
                        val_loader,
                        metrics,
                        criterion,
                        translator,
                        tracker=tracker,
                        use_cuda=use_cuda,
                    )
                    record_validation_stats(metric_values, loss, tracker, rank)
                    if tracker.goal_reached:
                        break

                    model.train()
                    tracker.train()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metric_values, loss = validation_round(
                val_loader,
                metrics,
                criterion,
                translator,
                tracker=tracker,
                use_cuda=use_cuda,
            )
            is_best = record_validation_stats(metric_values, loss, tracker, rank)
            checkpointer.save(
                tracker, model, optimizer, scheduler, tracker.current_epoch, is_best
            )
            tracker.epoch_end()

            if tracker.goal_reached:
                print("Goal Reached!")
                time.sleep(10)
                return
    else:
        cecf = CheckpointsEvaluationControlFlow(
            ckpt_dir=ckpt_run_dir,
            rank=rank,
            world_size=world_size,
            checkpointer=checkpointer,
            model=model,
            epochs=train_epochs,
            loss_function=criterion,
            metrics=metrics,
            use_cuda=use_cuda,
            dtype="fp32",
            max_batch_per_epoch=None,
        )

        train_stats = cecf.evaluate_by_epochs(train_loader)
        with open(os.path.join(output_dir, "train_stats.json"), "w") as f:
            json.dump(train_stats, f)

        # val_stats = cecf.evaluate_by_epochs(val_loader)
        # with open(os.path.join(output_dir, "val_stats.json"), "w") as f:
        #     json.dump(val_stats, f)


def main(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    rank,
    backend,
    hosts,
    validation_only=False,
    gpu=False,
    light_target=False,
):
    r"""Main logic."""
    with initialize_backends(
        comm_backend=backend,
        hosts=hosts,
        rank=rank,
        logging_level="INFO",
        logging_file=os.path.join(output_dir, "mlbench.log"),
        use_cuda=gpu,
        seed=43,
        cudnn_deterministic=False,
        ckpt_run_dir=ckpt_run_dir,
        delete_existing_ckpts=not validation_only,
    ):
        train_loop(
            run_id,
            dataset_dir,
            ckpt_run_dir,
            output_dir,
            validation_only,
            use_cuda=gpu,
            light_target=light_target,
            seed=43,
        )


if __name__ == "__main__":
    uid = args.uid
    dataset_dir = os.path.join(args.root_dataset, "torch", "wmt17")
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(
        args.run_id,
        dataset_dir,
        ckpt_run_dir,
        output_dir,
        rank=args.rank,
        backend=args.backend,
        hosts=args.hosts,
        validation_only=args.validation_only,
        gpu=args.gpu,
        light_target=args.light,
    )
