import torch
import torch.nn as nn

_MAX_WAVELENGTH = 10_000

class Rope(nn.Module):
    def __init__(self, head_dim: int, max_wavelength: int = _MAX_WAVELENGTH) -> None:
        super().__init__()

        fraction = 2 * torch.arange(0, head_dim // 2) / head_dim
        timescale = max_wavelength**fraction

        self.register_buffer("timescale", timescale)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # <could be cached>
        sinusoid_inp = positions[..., None] / self.timescale[None, None, :].to(inputs.dtype)
        sinusoid_inp = sinusoid_inp[..., None, :]
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        # </could be cached>

        first_half, second_half = torch.chunk(inputs, 2, -1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        out = torch.concatenate([first_part, second_part], -1)
        return out
