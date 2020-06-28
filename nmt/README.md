# Collaborative Multi-Head Attention: NMT Experiments

## Setup

Build the image in this folder

```
docker build . -t mlbench-core-collab
```

Run this docker image. If you are running in a kubernetess cluster, you should unset the following environment variable to avoid MLBench to try to set a distributed cluster:

```
unset  KUBERNETES_SERVICE_HOST
```

## Reproduce

We run on a single V100 machine for 6-12h. Run the following commands in `/codes`:

```bash
# original transformer with 16 concatenated heads of dim 64 (key_query_dim=1024)
python main.py --gpu --uid="original"
# smaller key_query_dim=128 using collaborative heads
python main.py --gpu --uid="collaborative128" --use_collaborative_heads --key_query_dim 128
# concatenate 16 heads of dim 8 (key_query_dim=1024)
python main.py --gpu --uid="concatenate128" --key_query_dim 128
```

We ran with the hyperparameters defined in the code and obtained the following results in the paper:

| $D_k$ | concat. | collab.
| ----- | ------- | -------
|  1024 | 27.8    |   -
|   256 | 27.7    |   27.8
|   128 | 27.4    |   27.8
|    64 | 26.6    |   27.3
|    32 | 25.5    |   27.1
