from .adapter_base import CollaborativeLayerAdapter
from .collaborative_attention import CollaborativeAttention


def albert_layers(model):
    if hasattr(model, "albert"):
        layers = model.albert.encoder.albert_layer_groups[0].albert_layers
    else:
        layers = model.encoder.albert_layer_groups[0].albert_layers
    return layers


class ALBERTCollaborativeAdapter(CollaborativeLayerAdapter):
    def __init__(self, layer):
        super().__init__(layer)

    @staticmethod
    def num_layers(model):
        return len(albert_layers(model))

    @staticmethod
    def get_layer(model, i):
        return albert_layers(model)[i].attention

    @staticmethod
    def set_layer(model, i, new_layer):
        albert_layers(model)[i].attention = new_layer

    @staticmethod
    def wrap_layer_args(layer):
        return layer

    @property
    def attention_probs_dropout_prob(self):
        return self.layer.dropout.p

    @property
    def dim_key_query_all(self):
        return self.layer.hidden_size

    @property
    def num_attention_heads(self):
        return self.layer.num_attention_heads

    @property
    def WQ(self):
        return self.layer.query.weight

    @property
    def bQ(self):
        return self.layer.query.bias

    @property
    def WK(self):
        return self.layer.key.weight

    @property
    def bK(self):
        return self.layer.key.bias

    @property
    def WV(self):
        return self.layer.value.weight

    @property
    def bV(self):
        return self.layer.value.bias

    @property
    def WO(self):
        return self.layer.dense.weight

    @property
    def bO(self):
        return self.layer.dense.bias

    @property
    def layerNorm(self):
        return self.layer.LayerNorm
