from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from einops import rearrange

from sentiment_lm_torch.model.positional_embeddings import apply_rope


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

        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.d_model,
            bias=False,
            dtype=self.dtype,
        )

    def forward(self, inputs: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor], block_mask: BlockMask) -> torch.Tensor:
        in_proj = self.in_proj(inputs)
        in_proj = rearrange(in_proj, "... seq (heads qkv) -> ... heads seq qkv", heads=self.num_heads)
        query, key, value = torch.chunk(in_proj, 3, -1)

        query = apply_rope(query, rope_cache)
        key = apply_rope(key, rope_cache)

        x = flex_attention(query, key, value, block_mask=block_mask)
        
        x = rearrange(x, "... heads seq qkv -> ... seq (heads qkv)")
        x = self.out_proj(x)

        return x

