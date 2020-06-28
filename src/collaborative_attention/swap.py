import tqdm

import torch
import tensorly as tl

tl.set_backend("pytorch")
from tensorly.decomposition import parafac

from .collaborative_attention import CollaborativeAttention, MixingMatrixInit


def swap_to_collaborative(model, adapter, dim_shared_query_key, tol=1e-6):
    print("Swap concatenate self-attention layers to collaborative...")
    for i in tqdm.trange(adapter.num_layers(model)):
        # plug the current layer into the adapter to have access to the fields we need
        layer = adapter(adapter.get_layer(model, i))

        # create the collaborative layer
        new_layer = CollaborativeAttention(
            dim_input=layer.dim_input,
            dim_value_all=layer.dim_value_all,
            dim_key_query_all=dim_shared_query_key,
            dim_output=layer.dim_output,
            num_attention_heads=layer.num_attention_heads,
            output_attentions=False,
            attention_probs_dropout_prob=layer.attention_probs_dropout_prob,
            use_dense_layer=layer.use_dense_layer,
            use_layer_norm=layer.use_layer_norm,
            mixing_initialization=MixingMatrixInit.CONCATENATE,
        )

        WK_per_head = layer.WK.view([layer.num_attention_heads, -1, layer.dim_input])

        if layer.dim_key_query_all != dim_shared_query_key:
            # tensor decomposition to get shared projections and mixing
            WQ_per_head = layer.WQ.view(
                [layer.num_attention_heads, -1, layer.dim_input]
            )
            WQWKT_per_head = torch.einsum("hdq,hdk->qhk", WQ_per_head, WK_per_head)

            # tensor decomposition
            _, factors = parafac(
                WQWKT_per_head.detach(), dim_shared_query_key, init="random", tol=tol
            )
            WQ_shared, mixing, WK_shared = factors
            new_layer.key.weight.data.copy_(WK_shared.transpose(0, 1))
            new_layer.query.weight.data.copy_(WQ_shared.transpose(0, 1))
            new_layer.mixing.data.copy_(mixing)

        else:
            # we simply copy the original matrices, mixing is initialized to concatenate
            new_layer.key.weight.data.copy_(layer.WK)
            new_layer.query.weight.data.copy_(layer.WQ)

        # bias reparametrization
        bq_per_head = layer.bQ.reshape([layer.num_attention_heads, -1])
        content_bias = bq_per_head.unsqueeze(1) @ WK_per_head
        content_bias = content_bias.squeeze(1)
        new_layer.content_bias.weight.data.copy_(content_bias)

        # value parameters are simply copied
        new_layer.value.weight.data.copy_(layer.WV)
        new_layer.value.bias.data.copy_(layer.bV)

        # copy output dense layer if exists
        if layer.use_dense_layer:
            new_layer.dense.weight.data.copy_(layer.WO)
            new_layer.dense.bias.data.copy_(layer.bO)

        # copy layernorm if exists
        if layer.use_layer_norm:
            new_layer.layer_norm.weight.data.copy_(layer.layerNorm.weight)
            new_layer.layer_norm.bias.data.copy_(layer.layerNorm.bias)

        new_layer = new_layer.to(layer.device)
        new_layer = adapter.wrap_layer_args(new_layer)
        adapter.set_layer(model, i, new_layer)
