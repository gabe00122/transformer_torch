from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask


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


def flip_head_length(x: torch.Tensor):
    return einops.rearrange(x, "... heads length depth -> ... length heads depth")


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

        self.rope = Rope()

        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.d_model,
            bias=False,
            dtype=self.dtype,
        )

    def forward(self, inputs: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor], block_mask: BlockMask) -> torch.Tensor:
        in_proj = self.in_proj(inputs)
        in_proj = einops.rearrange(in_proj, "... (heads qkv) -> ... heads qkv", heads=self.num_heads)
        query, key, value = torch.chunk(in_proj, 3, -1)

        query = self.rope(query, rope_cache)
        key = self.rope(key, rope_cache)

        query = flip_head_length(query)
        key = flip_head_length(key)
        value = flip_head_length(value)
        x = flex_attention(query, key, value, block_mask=block_mask)
        x = flip_head_length(x)
        
        x = einops.rearrange(x, "... heads qkv -> ... (heads qkv)")
        x = self.out_proj(x)

        return x

