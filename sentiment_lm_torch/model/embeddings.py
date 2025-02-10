import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedder(nn.Module):
    def __init__(self, vocab_size: int, embedding_features: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_features = embedding_features
    
        self.embedding_table = nn.Parameter(torch.randn(vocab_size, embedding_features) * 0.01)
    
    def forward(self, x: torch.Tensor, decode: bool) -> torch.Tensor:
        if decode:
            return self._decode(x)
        else:
            return self._encode(x)

    def _encode(self, indices: torch.Tensor) -> torch.Tensor:
        x = F.embedding(indices.to(torch.int), self.embedding_table)
        x *= math.sqrt(self.embedding_features)
        return x
    
    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.embedding_table.T)
