import torch
import torch.nn as nn
import torch.nn.functional as F

class FFBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        *,
        activation: nn.Module = nn.SiLU()
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_features = hidden_features

        self.activation = activation
        self.up_proj = nn.Linear(d_model, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class GLUBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        *,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_features = hidden_features
        self.activation = activation
        
        self.up_proj = nn.Linear(d_model, hidden_features * 2, bias=False)
        self.down_proj = nn.Linear(hidden_features, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)

        x, gate = torch.chunk(x, 2, -1)
        x = self.activation(x) * gate
        x = self.down_proj(x)
        return x
