import torch.nn as nn

from .adapter_base import CollaborativeLayerAdapter
from .collaborative_attention import CollaborativeAttention


class ArgsAdapterDistilBERT(nn.Module):
    def __init__(self, collab_layer):
        super().__init__()
        self.collab_layer = collab_layer

    def forward(self, query, key, value, mask, head_mask):
        return self.collab_layer(
            hidden_states=query,
            attention_mask=None,
            head_mask=head_mask,
            encoder_hidden_states=key,
            encoder_attention_mask=None,
        )


def distilbert_layers(model):
    if hasattr(model, "distilbert"):
        layers = model.distilbert.transformer.layer
    else:
        layers = model.transformer.layer
    return layers


class DistilBERTCollaborativeAdapter(CollaborativeLayerAdapter):
    def __init__(self, layer):
        super().__init__(layer)

    @staticmethod
    def num_layers(model):
        return len(distilbert_layers(model))

    @staticmethod
    def get_layer(model, i):
        return distilbert_layers(model)[i].attention

    @staticmethod
    def set_layer(model, i, new_layer):
        distilbert_layers(model)[i].attention = new_layer

    @staticmethod
    def wrap_layer_args(layer):
        return ArgsAdapterDistilBERT(layer)

    @property
    def attention_probs_dropout_prob(self):
        return self.layer.dropout.p

    @property
    def dim_key_query_all(self):
        return self.WK.shape[0]

    @property
    def num_attention_heads(self):
        return self.layer.n_heads

    @property
    def WQ(self):
        return self.layer.q_lin.weight

    @property
    def bQ(self):
        return self.layer.q_lin.bias

    @property
    def WK(self):
        return self.layer.k_lin.weight

    @property
    def bK(self):
        return self.layer.k_lin.bias

    @property
    def WV(self):
        return self.layer.v_lin.weight

    @property
    def bV(self):
        return self.layer.v_lin.bias

    @property
    def WO(self):
        return self.layer.out_lin.weight

    @property
    def bO(self):
        return self.layer.out_lin.bias

    @property
    def layerNorm(self):
        return None
