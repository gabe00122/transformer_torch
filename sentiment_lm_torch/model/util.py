import torch
from torch import nn

from sentiment_lm_torch.model.embeddings import Embedder


def name_to_activation(name: str) -> torch.nn.Module:
    """Convert a string to an activation function."""
    name = name.lower()

    match name:
        case "relu":
            return torch.nn.ReLU()
        case "gelu":
            return torch.nn.GELU()
        case "silu":
            return torch.nn.SiLU()
        case _:
            raise ValueError(f"Unknown activation function: {name}")


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, Embedder):
        nn.init.normal_(module.embedding_table, std=0.01)
