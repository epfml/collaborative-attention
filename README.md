# Collaborative Attention

Code for the paper [Multi-Head Attention: Collaborate Instead of Concatenate](https://arxiv.org/abs/2006.16362), Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi.

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

| Model              | $\tilde D_k$        | CoLA | SST-2 | MRPC      | STS-B     | QQP       | MNLI      | QNLI | RTE  | **Avg.**
| ------------------ | ------------------- | ---- | ----- | --------- | --------- | --------- | --------- | ---- | ---- | --------
| BERT-base          |  -                  | 54.7 | 91.7  | 88.8/83.8 | 88.8/88.7 | 87.6/90.8 | 84.1      | 90.9 | 63.2 |  83.0
|                    |  768                | 56.8 | 90.1  | 89.6/85.1 | 89.2/88.9 | 86.8/90.2 | 83.4      | 90.2 | 65.3 |  83.2
|                    | 384                 | 56.3 | 90.7  | 87.7/82.4 | 88.3/88.0 | 86.3/90.0 | 83.0      | 90.1 | 65.3 |  82.5
|                    | 256                 | 52.6 | 90.1  | 88.1/82.6 | 87.5/87.2 | 85.9/89.6 | 82.7      | 89.5 | 62.5 |  81.7
|                    | 128                 | 43.5 | 89.5  | 83.4/75.2 | 84.5/84.3 | 81.1/85.8 | 79.4      | 86.7 | 60.7 |  77.6
| DistilBERT         | -                   | 46.6 | 89.8  | 87.0/82.1 | 84.0/83.7 | 86.2/89.8 | 81.9      | 88.1 | 60.3 |  80.0
|                    | 384                 | 45.6 | 89.2  | 86.6/80.9 | 81.7/81.9 | 86.1/89.6 | 81.1      | 87.0 | 60.7 |  79.1
| ALBERT             | -                   | 58.3 | 90.7  | 90.8/87.5 | 91.2/90.8 | 87.5/90.7 | 85.2      | 91.7 | 73.7 |  85.3
|                    | 512                 | 51.1 | 86.0  | 91.4/88.0 | 88.6/88.2 | 87.2/90.4 | 84.2      | 90.2 | 69.0 |  83.1
|                    | 384                 | 40.7 | 89.6  | 82.3/71.1 | 86.0/85.6 | 87.2/90.5 | 84.4      | 90.0 | 49.5 |  77.9


### Neural Machine Translation

The NMT experiments is based on [MLBench](https://mlbench.readthedocs.io/).
You can reproduce our results using the [nmt/](nmt/) folder in this repository.

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
