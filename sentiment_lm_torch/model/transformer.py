import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from sentiment_lm_torch.model.attention import AttentionBlock
from sentiment_lm_torch.model.embeddings import Embedder
from sentiment_lm_torch.model.feed_forward import FFBlock, GLUBlock


class TransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        *,
        activation: nn.Module = nn.SiLU(),
        glu: bool = True,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.ffn_size = ffn_size
        self.activation = activation
        self.glu = glu
        self.dtype = dtype

        self.attention_norm = nn.RMSNorm(d_model, dtype=dtype)
        self.attention = AttentionBlock(num_heads, d_model, dtype=dtype)

        # should it normalize over just the last dimension or the sequence length as well?
        self.ffn_norm = nn.RMSNorm(d_model, dtype=dtype)
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(d_model, ffn_size, activation=activation)

    def forward(self, x: Tensor, positions: Tensor, block_mask: BlockMask | None = None) -> Tensor:
        attention_input = self.attention_norm(x)
        attention = self.attention(attention_input, positions, block_mask=block_mask)
        x = x + attention

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = x + feed_forward

        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        *,
        activation: nn.Module = nn.SiLU(),
        glu: bool = True,
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.ffn_size = ffn_size
        
        self.activation = activation
        self.glu = glu
        self.dtype = dtype

        self.embedder = Embedder(vocab_size, d_model)

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerLayer(
                    num_heads,
                    d_model,
                    ffn_size,
                    activation=self.activation,
                    glu=glu,
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.output_norm = nn.RMSNorm(d_model, dtype=dtype)
    
    def init_kv_cache(self, batch_size: int, context_size: int, device: torch.device, dtype: torch.dtype):
        for layer in self.layers:
            layer.attention.init_kv_cache(batch_size, context_size, device, dtype)

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.clear_kv_cache()

    def forward(self, inputs: Tensor, positions: Tensor, block_mask: BlockMask | None = None) -> Tensor:
        x = self.embedder(inputs, decode=False)

        for layer in self.layers:
            x = layer(x, positions, block_mask=block_mask)

        x = self.output_norm(x)
        x = self.embedder(x, decode=True)
        # x = jnp.asarray(x, dtype=jnp.float32)

        return x
