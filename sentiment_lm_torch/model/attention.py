from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


import einops
import math

from sentiment_lm_torch.model.positional_embeddings import Rope


def _einsum_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
    attn_weights = torch.einsum("...qhd,...khd->...hqk", query, key)

    big_neg = torch.finfo(attn_weights.dtype).min
    attn_weights = torch.where(mask, attn_weights, big_neg)

    attn_weights = F.softmax(attn_weights, -1)

    return torch.einsum("...hqk,...khd->...qhd", attn_weights, value)

@lru_cache
def causal_block_mask(seq_len: int):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
    return block_mask


class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        use_flex_attention: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dtype = dtype
        self.use_flex_attention = use_flex_attention

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        self.in_proj = nn.Linear(
            self.d_model,
            self.num_heads * self.head_dim * 3,
            bias=False,
            dtype=self.dtype,
        )

        self.rope = Rope(head_dim=self.head_dim)

        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.d_model,
            bias=False,
            dtype=self.dtype,
        )

        _sqrt_depth = torch.tensor(math.sqrt(self.head_dim), dtype=self.dtype)
        self.register_buffer("_sqrt_depth", _sqrt_depth)

    def forward(self, inputs: torch.Tensor, segment_positions: torch.Tensor, mask: torch.Tensor):
        in_proj = self.in_proj(inputs)
        in_proj = einops.rearrange(in_proj, "... (heads qkv) -> ... heads qkv", heads=self.num_heads)
        query, key, value = torch.chunk(in_proj, 3, -1)

        query = self.rope(query, segment_positions)
        key = self.rope(key, segment_positions)

        if self.use_flex_attention:
            # def causal_mask(score, b, h, q_idx, kv_idx):
            #     return torch.where(q_idx >= kv_idx, score, -float("inf"))
            block_mask = causal_block_mask(query.shape[-3]) # query.shape[-3]
            
            query = einops.rearrange(query, "... l h q -> ... h l q")
            key = einops.rearrange(key, "... l h q -> ... h l q")
            value = einops.rearrange(value, "... l h q -> ... h l q")
            x = flex_attention(query, key, value, block_mask=block_mask)
            x = einops.rearrange(x, "... h l q -> ... l h q")
        else:
            query = query / self._sqrt_depth
            x = _einsum_attention(query, key, value, mask)
        
        # x = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        # print(x.shape)
        x = einops.rearrange(x, "... heads qkv -> ... (heads qkv)")

        x = self.out_proj(x)

        return x

