import torch.nn as nn


class CollaborativeLayerAdapter:
    def __init__(self, layer):
        self.layer = layer

    @staticmethod
    def num_layers(model):
        raise NotImplemented

    @staticmethod
    def get_layer(model, i):
        raise NotImplemented

    @staticmethod
    def set_layer(model, i, new_layer):
        raise NotImplemented

    @staticmethod
    def wrap_layer_args(layer):
        return layer

    """
    The following methods tell where are the layer parameters needed for reparametrization.
    """

    @property
    def num_attention_heads(self):
        raise NotImplemented

    @property
    def attention_probs_dropout_prob(self):
        raise NotImplemented

    @property
    def dim_key_query_all(self):
        raise NotImplemented

    @property
    def WQ(self):
        raise NotImplemented

    @property
    def bQ(self):
        raise NotImplemented

    @property
    def WK(self):
        raise NotImplemented

    @property
    def bK(self):
        raise NotImplemented

    @property
    def WV(self):
        raise NotImplemented

    @property
    def bV(self):
        raise NotImplemented

    @property
    def WO(self):
        raise NotImplemented

    @property
    def bO(self):
        raise NotImplemented

    @property
    def layerNorm(self):
        raise NotImplemented

    """
    You should not have to override this
    """

    @property
    def dim_input(self):
        return self.WQ.shape[1]

    @property
    def dim_output(self):
        return self.WO.shape[0] if self.use_dense_layer else self.WV.shape[0]

    @property
    def device(self):
        return self.WQ.device

    @property
    def dim_value_all(self):
        return self.WV.shape[0]

    @property
    def use_dense_layer(self):
        return self.WO is not None

    @property
    def use_layer_norm(self):
        return self.layerNorm is not None
