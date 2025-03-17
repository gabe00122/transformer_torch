import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from sentiment_lm_torch.model.attention import AttentionBlock
from sentiment_lm_torch.model.embeddings import Embedder
from sentiment_lm_torch.model.feed_forward import FFBlock, GLUBlock

class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        The GRUGatingUnit module implements the GRU gating mechanism used in the GTrXL model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Overview:
            Initialize the GRUGatingUnit module.
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input.
            - bg (:obj:`bg`): The gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to \
                be close to the identity map. This can greatly improve the learning speed and stability since it \
                initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """

        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute the output value using the GRU gating mechanism.
        Arguments:
            - x: (:obj:`torch.Tensor`): The first input tensor.
            - y: (:obj:`torch.Tensor`): The second input tensor. \
                x and y should have the same shape and their last dimension should match the input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): The output of the GRU gating mechanism. \
                The shape of g matches the shapes of x and y.
        """

        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape

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

        self.attention_norm = nn.LayerNorm(d_model, dtype=dtype)
        self.attention = AttentionBlock(num_heads, d_model, dtype=dtype)

        # should it normalize over just the last dimension or the sequence length as well?
        self.ffn_norm = nn.LayerNorm(d_model, dtype=dtype)
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(d_model, ffn_size, activation=activation)

        # self.attn_gate = GRUGatingUnit(d_model)
        # self.ffn_gate = GRUGatingUnit(d_model)

    def forward(self, x: Tensor, positions: Tensor, block_mask: BlockMask | None = None) -> Tensor:
        attention_input = self.attention_norm(x)
        attention = self.attention(attention_input, positions, block_mask=block_mask)
        x = x + attention# * self.attn_gate(x)
        # x = self.attn_gate(x, attention)

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = x + feed_forward# * self.ffn_gate(x)
        # x = self.ffn_gate(x, feed_forward)

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
