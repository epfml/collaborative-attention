# Code from HuggingFace

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
import time
from typing import Dict, Optional

import numpy as np
import torch
# import wandb

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from collaborative_attention import (
    swap_to_collaborative,
    BERTCollaborativeAdapter,
    DistilBERTCollaborativeAdapter,
    ALBERTCollaborativeAdapter,
)
import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    mix_heads: bool = False
    mix_size: Optional[int] = None  # Size of the tensor decomposition for mixed heads
    mix_decomposition_tol: float = 1e-6  # Tolerance for the tensor factorization algorithm
    repeat_id: int = 0  # Useless arg to do multiple run of a same config in wandb sweep
    model_output_prefix: Optional[str] = None

    # context / content only attention
    # made weird to allow grid search in wandb
    restricted_attention: bool = False
    context_attention_only: int = -1  # 1 -> context attention
    # 0 -> content attention


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb.init(project="mixing-heads-finetuning")
    # wandb.config.update(model_args)
    # wandb.config.update(data_args)
    # wandb.config.update(training_args)

    # HERE MODIFY THE CONFIG PATHS ...
    def extract_last(path):
        return [s for s in path.split("/") if s][-1]

    restricted_prefix = ""
    if model_args.restricted_attention:
        if model_args.context_attention_only == 1:
            restricted_prefix = "context_only-"
        elif model_args.context_attention_only == 0:
            restricted_prefix = "content_only-"
        else:
            raise ValueError("Should set context_attention_only to 0 or 1")

    output_model_name = (
        (model_args.model_output_prefix or "")
        + ("finetuned-" if training_args.do_train else "")
        + ("mix{}-".format(model_args.mix_size) if model_args.mix_size else "")
        + restricted_prefix
        + extract_last(model_args.model_name_or_path)
    )

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        output_model_name,
        data_args.task_name,
        str(model_args.repeat_id),
    )

    data_args.data_dir = os.path.join(data_args.data_dir, data_args.task_name)

    if training_args.num_train_epochs == 3.0 and data_args.task_name.lower() in [
        "sst-2",
        "rte",
    ]:
        training_args.num_train_epochs = 10.0
        print(
            "OVERIDE NUMBER OF EPOCH FOR TASK {} TO {}".format(
                data_args.task_name, training_args.num_train_epochs
            )
        )

    if os.path.exists(model_args.model_name_or_path):
        model_args.model_name_or_path = os.path.join(
            model_args.model_name_or_path,
            data_args.task_name,
            str(model_args.repeat_id),
        )

    # DONE MODIFYING THE CONFIG

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(
        model_args.repeat_id if model_args.repeat_id is not None else training_args.seed
    )

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    if model_args.restricted_attention and model_args.mix_heads:
        raise ValueError(
            "Context/content attention and mix heads not implemented together correctly"
        )

    if model_args.restricted_attention:
        if hasattr(model, "bert"):
            layers = model.bert.encoder.layer
        elif hasattr(model, "electra"):
            layers = model.electra.encoder.layer
        else:
            raise Exception(
                'Does not support transforming model "{}" to mixed self-attention.'.format(
                    type(model)
                )
            )

        print(
            "Make {}-only self-attention layers...".format(
                "context" if model_args.context_attention_only == 1 else "content"
            )
        )
        for i in tqdm.trange(len(layers)):
            # set b_K = 0
            layers[i].attention.self.key.bias.requires_grad = False
            layers[i].attention.self.key.bias.zero_()

            if model_args.context_attention_only == 1:
                # set b_Q = 0
                layers[i].attention.self.query.bias.requires_grad = False
                layers[i].attention.self.query.bias.zero_()
            else:  # content attention only
                # set W_Q = 0
                layers[i].attention.self.query.weight.requires_grad = False
                layers[i].attention.self.query.weight.zero_()

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    if model_args.mix_heads:
        start = time.time()

        adapter = BERTCollaborativeAdapter
        if "albert" in model_args.model_name_or_path.lower():
            adapter = ALBERTCollaborativeAdapter
        if "distilbert" in model_args.model_name_or_path.lower():
            adapter = DistilBERTCollaborativeAdapter

        swap_to_collaborative(
            model,
            adapter,
            dim_shared_query_key=model_args.mix_size,
            tol=model_args.mix_decomposition_tol,
        )

        elapsed = time.time() - start
        # wandb.run.summary["decomposition_time"] = elapsed

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True)
            )

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{eval_dataset.args.task_name}.txt",
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info(
                        "***** Eval results {} *****".format(
                            eval_dataset.args.task_name
                        )
                    )
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        # wandb.run.summary[key] = value

            results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
