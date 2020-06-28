# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# Modification of the original code to handle collaborative multi-head attention
# See the paper "Multi-Head Attention: Collaborate Instead of Concateane" for
# more details. You can check the small changes we made running a diff with
# the file mlbench-core/models/pytorch/transformers/modules/multihead_attention.py

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.variable import Variable
from torch.nn import Parameter

from mlbench_core.models.pytorch.transformer.modules import strided_batched_gemm


class QueryLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_q, scale):
        s = Variable(torch.tensor([scale]))
        ctx.save_for_backward(input, weights_q, s)
        q = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_q,
            beta=0.0,
            alpha=s[0],
        )
        q = q.view(input.size(0), input.size(1), q.size(-1))
        return q.detach()

    @staticmethod
    def backward(ctx, q_grad):
        input, weights_q, s = ctx.saved_tensors
        input = input.view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)
        q = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)),
            weights_q.transpose(0, 1),
            beta=0.0,
            alpha=s[0],
        )
        q = q.view(q_grad.size(0), q_grad.size(1), input.size(0))
        q_grad = q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2))
        weights_q_grad = torch.addmm(weights_q, input, q_grad, beta=0.0, alpha=s[0])
        return q, weights_q_grad, None


class KeyValueLinears(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_k, weights_v):
        ctx.save_for_backward(input, weights_k, weights_v)
        k = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_k,
            beta=0.0,
            alpha=1.0,
        )
        k = k.view(input.size(0), input.size(1), k.size(-1))
        v = torch.addmm(
            input.view(input.size(0) * input.size(1), input.size(2)),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_v,
            beta=0.0,
            alpha=1.0,
        )
        v = v.view(input.size(0), input.size(1), input.size(2))
        return k.detach(), v.detach()

    @staticmethod
    def backward(ctx, k_grad, v_grad):
        input, weights_k, weights_v = ctx.saved_tensors
        input = input.view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)
        k = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)),
            weights_k.transpose(0, 1),
            beta=0.0,
        )
        k_grad = k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(
            v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)),
            weights_v.transpose(0, 1),
            beta=1.0,
        )
        v = v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_k_grad, weights_v_grad


class SelfAttentionLinears(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_q, weights_k, weights_v, scale):
        s = Variable(torch.tensor([scale]))
        ctx.save_for_backward(input, weights_q, weights_k, weights_v, s)
        q = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_q,
            beta=0.0,
            alpha=s[0],
        )
        q = q.view(input.size(0), input.size(1), q.size(-1))
        k = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_k,
            beta=0.0,
            alpha=1.0,
        )
        k = k.view(input.size(0), input.size(1), k.size(-1))
        v = torch.addmm(
            input.view(input.size(0) * input.size(1), input.size(2)),
            input.view(input.size(0) * input.size(1), input.size(2)),
            weights_v,
            beta=0.0,
            alpha=1.0,
        )
        v = v.view(input.size(0), input.size(1), input.size(2))
        return q.detach(), k.detach(), v.detach()

    @staticmethod
    def backward(ctx, q_grad, k_grad, v_grad):
        input, weights_q, weights_k, weights_v, s = ctx.saved_tensors
        input = input.view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)
        q = torch.addmm(
            torch.zeros(1, device=input.device, dtype=input.dtype),
            q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)),
            weights_q.transpose(0, 1),
            beta=0.0,
            alpha=s[0],
        )
        q_grad = q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2))
        weights_q_grad = torch.addmm(weights_q, input, q_grad, beta=0.0, alpha=s[0])
        k = q.addmm_(
            k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)),
            weights_k.transpose(0, 1),
            beta=1.0,
        )
        k_grad = k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(
            v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)),
            weights_v.transpose(0, 1),
            beta=1.0,
        )
        v = v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_q_grad, weights_k_grad, weights_v_grad, None


class StridedBmm1Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = torch.empty(
            (input1.size(0), input1.size(1), input2.size(2)),
            dtype=input1.dtype,
            device=torch.device("cuda"),
        )
        if (input1.dtype == torch.float16) and (input2.dtype == torch.float16):
            output = strided_batched_gemm.strided_batched_gemm(
                0.0, output, 1.0, input1, input2
            )
        else:
            output = torch.bmm(input1, input2, out=output)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.empty(
            (input1.size(1), input2.size(0), input1.size(2)),
            dtype=input1.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)
        grad_input2 = torch.empty(
            (input2.size(2), input2.size(0), input2.size(1)),
            dtype=input2.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)
        if (
            (grad_output.dtype == torch.float16)
            and (input1.dtype == torch.float16)
            and (input2.dtype == torch.float16)
        ):
            grad_input1 = strided_batched_gemm.strided_batched_gemm(
                0.0, grad_input1, 1.0, grad_output, input2.transpose(1, 2)
            )
            grad_input2 = strided_batched_gemm.strided_batched_gemm(
                0.0, grad_input2, 1.0, grad_output.transpose(1, 2), input1
            )
            grad_input2 = grad_input2.transpose(1, 2)
        else:
            grad_input1 = torch.bmm(
                grad_output, input2.transpose(1, 2), out=grad_input1
            )
            grad_input2 = torch.bmm(
                grad_output.transpose(1, 2), input1, out=grad_input2
            ).transpose(1, 2)
        return grad_input1, grad_input2


class StridedBmm2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = torch.empty(
            (input1.size(1), input1.size(0), input2.size(2)),
            dtype=input1.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)
        if (input1.dtype == torch.float16) and (input2.dtype == torch.float16):
            output = strided_batched_gemm.strided_batched_gemm(
                0.0, output, 1.0, input1, input2
            )
        else:
            output = torch.bmm(input1, input2, out=output)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input2 = torch.empty(
            (input2.size(1), input2.size(0), input2.size(2)),
            dtype=input2.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)
        grad_input1 = torch.empty(
            (input1.size(0), input1.size(1), input1.size(2)),
            dtype=input2.dtype,
            device=torch.device("cuda"),
        )
        if (
            (grad_output.dtype == torch.float16)
            and (input1.dtype == torch.float16)
            and (input2.dtype == torch.float16)
        ):
            grad_input1 = strided_batched_gemm.strided_batched_gemm(
                0.0, grad_input1, 1.0, grad_output, input2.transpose(1, 2)
            )
            grad_input2 = strided_batched_gemm.strided_batched_gemm(
                0.0, grad_input2, 1.0, input1.transpose(1, 2), grad_output
            )
        else:
            grad_input1 = torch.bmm(grad_output, input2.transpose(1, 2))
            grad_input2 = torch.bmm(
                input1.transpose(1, 2), grad_output, out=grad_input2
            )
        return grad_input1, grad_input2


class FillPadding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, num_heads):
        assert len(inputs.size()) == 3, "Tensor is not 3 dims"
        bsz = int(inputs.size(0) / num_heads)
        tgt_len = inputs.size(1)
        src_len = inputs.size(2)
        inputs = inputs.view(bsz, num_heads, tgt_len, src_len)
        inputs = inputs + mask.unsqueeze(1).unsqueeze(2)
        inputs = inputs.view(bsz * num_heads, tgt_len, src_len)
        return inputs.detach()

    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None


query_linear = QueryLinear.apply
key_value_linears = KeyValueLinears.apply
self_attn_linears = SelfAttentionLinears.apply
strided_bmm1 = StridedBmm1Func.apply
strided_bmm2 = StridedBmm2Func.apply
fill_padding_func = FillPadding.apply

key_query_dim_default = None
use_collaborative_heads_default = False


class CollaborativeMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need"
    and "Multi-Head Attention: Collaborate Instead of Concatenate"
    for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        softmax_type="default",
        bias=False,
        key_query_dim=None,
        use_collaborative_heads=False,
    ):
        super().__init__()
        print(
            "Using CollaborativeMultiheadAttention with {} heads of total key_query_dim={}".format(
                "collaborative" if use_collaborative_heads else "concatenated",
                key_query_dim or embed_dim,
            )
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.softmax_type = softmax_type
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        self.key_query_dim = key_query_dim or embed_dim
        self.use_collaborative_heads = use_collaborative_heads

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, self.key_query_dim))
        self.in_proj_weight_k = Parameter(torch.Tensor(embed_dim, self.key_query_dim))
        self.in_proj_weight_v = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias_q = Parameter(torch.Tensor(self.key_query_dim))
            self.in_proj_bias_k = Parameter(torch.Tensor(self.key_query_dim))
            self.in_proj_bias_v = Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter("in_proj_bias_k", None)
            self.register_parameter("in_proj_bias_q", None)
            self.register_parameter("in_proj_bias_v", None)

        if self.use_collaborative_heads:
            self.mixing = Parameter(torch.Tensor(num_heads, self.key_query_dim))
        else:
            self.register_parameter("mixing", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight_q)
        nn.init.xavier_uniform_(self.in_proj_weight_k)
        nn.init.xavier_uniform_(self.in_proj_weight_v)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.use_collaborative_heads:
            nn.init.xavier_uniform_(self.mixing)

        if self.in_proj_bias_k is not None:
            nn.init.constant_(self.in_proj_bias_q, 0.0)
            nn.init.constant_(self.in_proj_bias_k, 0.0)
            nn.init.constant_(self.in_proj_bias_v, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        mask_future_timesteps=False,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
    ):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self_attn_linears(
                query,
                self.in_proj_weight_q,
                self.in_proj_weight_k,
                self.in_proj_weight_v,
                self.scaling,
            )
        elif kv_same:
            # encoder-decoder attention
            q = query_linear(query, self.in_proj_weight_q, self.scaling)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = key_value_linears(
                    key, self.in_proj_weight_k, self.in_proj_weight_v
                )
        else:

            q = torch.addmm(
                query.view(query.size(0) * query.size(1), query.size(2)),
                query.view(query.size(0) * query.size(1), query.size(2)),
                self.in_proj_weight_q,
                beta=0.0,
                alpha=self.scaling,
            )
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k = F.linear(key, self.in_proj_weight_k, self.in_proj_bias_k)
                v = F.linear(value, self.in_proj_weight_v, self.in_proj_bias_v)

        if saved_state is not None:
            if "prev_key" in saved_state:
                k = torch.cat((saved_state["prev_key"], k), dim=0)
            if "prev_value" in saved_state:
                v = torch.cat((saved_state["prev_value"], v), dim=0)
            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.use_collaborative_heads:
            q = (
                (
                    (q.contiguous().reshape(tgt_len, bsz, 1, self.key_query_dim))
                    * self.mixing.reshape(1, 1, self.num_heads, self.key_query_dim)
                )
                .reshape(tgt_len, bsz * self.num_heads, self.key_query_dim)
                .transpose(0, 1)
            )
            k = (
                k.contiguous()
                .reshape(src_len, bsz, 1, self.key_query_dim)
                .repeat(1, 1, self.num_heads, 1)
                .reshape(src_len, bsz * self.num_heads, self.key_query_dim)
                .transpose(0, 1)
            )

            attn_weights = strided_bmm1(q, k.transpose(1, 2))
        else:
            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads, q.size(-1) // self.num_heads)
                .transpose(0, 1)
            )
            k = (
                k.contiguous()
                .view(src_len, bsz * self.num_heads, k.size(-1) // self.num_heads)
                .transpose(0, 1)
            )

            attn_weights = strided_bmm1(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        v = (
            v.contiguous()
            .view(src_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert (
                query.size() == key.size()
            ), "mask_future_timesteps only applies to self-attention"
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)

        if self.softmax_type == "fast_fill":
            if key_padding_mask is not None:
                attn_weights = fill_padding_func(
                    attn_weights, key_padding_mask, self.num_heads
                )
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.softmax_type == "default":
            if key_padding_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            assert False, "Unknown Softmax Type: {}".format(self.softmax_type)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = strided_bmm2(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, "attn_state") or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, "attn_state", buffer)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, "_mlbench_instance_id"):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._mlbench_instance_id = INCREMENTAL_STATE_INSTANCE_ID[
            module_name
        ]

    return "{}.{}.{}".format(module_name, module_instance._mlbench_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value
