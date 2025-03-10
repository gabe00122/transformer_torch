from functools import lru_cache

import torch
from torch import nn, Tensor

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from einops import rearrange

from sentiment_lm_torch.model.positional_embeddings import apply_rope


def causal_block_mask(seq_len: int):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
    return block_mask


def attention(query: Tensor, key: Tensor, value: Tensor, block_mask: BlockMask | None) -> Tensor:
    key = rearrange(key, "... seq heads d -> ... heads seq d")
    query = rearrange(query, "... seq heads d -> ... heads seq d")
    value = rearrange(value, "... seq heads d -> ... heads seq d")
    
    out = flex_attention(query, key, value, block_mask=block_mask)
    out = rearrange(out, "... heads seq qkv -> ... seq (heads qkv)")
    return out


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
    
    def init_kv_cache(self, batch_size: int, context_size: int, device: torch.device, dtype: torch.dtype):
        shape = (batch_size, context_size, self.num_heads, self.head_dim)
        key_cache = torch.zeros(shape, device=device, dtype=dtype)
        value_cache = torch.zeros(shape, device=device, dtype=dtype)

        self.register_buffer("key_cache", key_cache, persistent=False)
        self.register_buffer("value_cache", value_cache, persistent=False)
        self.has_kv_cache = True
    
    def clear_kv_cache(self):
        self.key_cache.zero_()
        self.value_cache.zero_()

    def update_kv_cache(self, positions: Tensor, key: Tensor, value: Tensor):
        batch_idx = torch.arange(self.key_cache.shape[0], device=positions.device, dtype=torch.int64)
        self.key_cache[batch_idx, positions] = key
        self.value_cache[batch_idx, positions] = value

        return self.key_cache, self.value_cache

    def forward(self, inputs: torch.Tensor, positions: Tensor, block_mask: BlockMask | None = None) -> torch.Tensor:        
        in_proj = self.in_proj(inputs)
        in_proj = rearrange(in_proj, "... seq (heads qkv) -> ... seq heads qkv", heads=self.num_heads)
        query, key, value = torch.chunk(in_proj, 3, -1)

        query = apply_rope(query, positions)
        key = apply_rope(key, positions)

        if self.has_kv_cache and not self.training:
            key, value = self.update_kv_cache(positions, key, value)

        x = attention(query, key, value, block_mask=block_mask)
        x = self.out_proj(x)

        return x

