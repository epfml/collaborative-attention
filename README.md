# Collaborative Attention

Code for the paper [Multi-Head Attention: Collaborate Instead of Concatenate](https://arxiv.org/abs/2006.16362), Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi.

> Clone this repo with submodules `git clone --recurse-submodules https://github.com/epfml/collaborative-attention.git`

We provide a python package to reparametrize any pretrained attention layer into a collaborative attention layer.
This allows to decrease the key/query dimension without affecting the performance of the model.
Our factorization can be used either for pretraining as a drop-in replacement of concatenated heads attention or before fine tuning as a compression method.

[![tests](https://github.com/epfml/collaborative-attention/workflows/tests/badge.svg)](https://github.com/epfml/collaborative-attention/actions?query=workflow%3Atests)

## Install

Clone this repository and install the package with pip:

```bash
# you need to have PyTorch installed
git clone https://github.com/epfml/collaborative-attention.git
pip install -U -e collaborative-attention
```

## Quick Start

We provide code to reparametrize any attention layer into our efficient collaborative version.
The following code factorize a pretrained BERT-base with collaborative heads.

```python
from transformers import AutoModel
from collaborative_attention import swap_to_collaborative, BERTCollaborativeAdapter
import copy
import torch

model = AutoModel.from_pretrained("bert-base-cased-finetuned-mrpc")

# reparametrize the model with tensor decomposition to use collaborative heads
# decrease dim_shared_query_key to 384 for example to compress the model
collab_model = copy.deepcopy(model)
swap_to_collaborative(collab_model, BERTCollaborativeAdapter, dim_shared_query_key=768)

# check that output is not altered too much
any_input = torch.LongTensor(3, 25).random_(1000, 10000)
collab_model.eval()  # to disable dropout
out_collab = collab_model(any_input)

model.eval()
out_original = model(any_input)

print("Max l1 error: {:.1e}".format((out_collab[0] - out_original[0]).abs().max().item()))
# >>> Max l1 error: 1.9e-06

# You can evaluate the new model, refine tune it or save it.
# We also want to pretrain our collaborative head from scratch (if you were wondering).
```

## Explore the Code

- The collaborative multi-head attention layer is defined in [src/collaborative_attention/collaborative_attention.py](src/collaborative_attention/collaborative_attention.py).
- We use [tensorly](http://tensorly.org/stable/index.html) to decompose a trained attention head and reparametrize it as a collaborative layer. You can look at the decomposition code in [src/collaborative_attention/swap.py](src/collaborative_attention/swap.p) that defines the `swap_to_collaborative` function.
When run on a GPU, the decomposition takes less than a minute per layer.

## Other Transformers

Our framework can be adapted on any transformer that we know of.
Our code base is modular so that we can swap collaborative heads in any transformer.
We use small adapter classes that extract the parameters of the layers we want to transform.
We have defined adapters for the following transformers:

| Model | Adapter Class | File |
| ----- | ------------- | ---- |
| [BERT](https://arxiv.org/abs/1810.04805) | BERTCollaborativeAdapter | `src/collaborative_attention/adapter_bert.py` |
| [DistilBERT](https://arxiv.org/abs/1910.01108) | DistilBERTCollaborativeAdapter | `src/collaborative_attention/adapter_distilbert.py` |
| [ALBERT](https://arxiv.org/abs/1909.11942) | ALBERTCollaborativeAdapter | `src/collaborative_attention/adapter_albert.py` |

Adding a new model is very simple: define your own adapter based on `CollaborativeLayerAdapter`. You simply have to write a few one liner functions and you can get inspiration from the files above. We are happy to quickly merge PR, just copy paste a test in `tests/` to make sure your adapter is working.

## Results

### Natural Language Understanding

Download the GLUE data following [this](https://github.com/huggingface/transformers/tree/master/examples/text-classification) and set `GLUE_DIR` environment variable.

You should proceed in two steps
1. Fine tune the original model, `bert-base-cased` for example, for the task (without `--mix_heads` and `--mix_size`)
2. Use the saved finetuned model in `output/` to do the decomposition (`model_name_or_path` argument), it will swap it to collaborative and re-finetune.

We show a comand example with an already finetuned model on MRPC:

```
python run_glue.py \
    --model_name_or_path=bert-base-cased-finetuned-mrpc \
    --task_name=mrpc \
    --data_dir=$GLUE_DIR \
    --output_dir=output/ \
    --do_train \
    --do_eval \
    --max_seq_length=128 \
    --per_gpu_train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --overwrite_output_dir \
    --save_total_limit=3 \
    --mix_heads \
    --mix_size 384
```

### Neural Machine Translation

```
cd fairseq/
pip install --editable ./
# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

Download and preprocess the data following these [instructions](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt).

Reproduce our experiments on a machine with 4 GPUs with the following command:

```bash
# set COLAB to "none" to run the original transformer
# set KEY_DIM for different key dimensions
KEY_DIM=512 COLAB="encoder_cross_decoder" CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/wmt16_en_de_bpe32k \
    --arch transformer_wmt_en_de \
    --save-dir checkpoints/wmt16-en-de/base-d-$KEY_DIM-colab-$COLAB \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0007 \
    --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 3584 \
    --update-freq 2 \
    --fp16 \
    --collaborative-heads $COLAB \
    --key-dim $KEY_DIM \
```

### Vision Transformers

Follow deit setup

```
cd deit
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2 tensorly
```

To train Base3 models, run the following command:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_base3_patch16_224_collab384 --batch-size 256 --data-path /imagenet --output_dir ../outputs
```

or for the concatenate attention:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_base3_patch16_224_key384 --batch-size 256 --data-path /imagenet --output_dir ../outputs
```

You can reparametrize a pretrained model by running the following command on a single GPU machine:

```
python --model deit_base_patch16_224 --shared_key_query_dim 384 --output_dir ./models
```

which will create a new checkpoint for this reparametrized model in `./models/deit_base_patch16_224_collab384.pt`.

To evaluate this model, run:

```
python main.py --eval --model deit_base_patch16_224_collab384 --data-path /imagenet --pretrained  --models_directory ./models
```


## Citation

If you find this code useful, please cite the paper:

```
@misc{cordonnier2020multihead,
    title={Multi-Head Attention: Collaborate Instead of Concatenate},
    author={Jean-Baptiste Cordonnier and Andreas Loukas and Martin Jaggi},
    year={2020},
    eprint={2006.16362},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
