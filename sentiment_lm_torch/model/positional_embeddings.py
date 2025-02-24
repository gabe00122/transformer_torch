import torch
from torch import Tensor


_MAX_WAVELENGTH = 10_000

# def create_rope_cache(head_dim: int, positions: torch.Tensor, max_wavelength: int = _MAX_WAVELENGTH) -> tuple[torch.Tensor, torch.Tensor]:
#     fraction = 2 * torch.arange(0, head_dim // 2) / head_dim
#     timescale = max_wavelength**fraction

#     sinusoid_inp = positions[..., None] / timescale[None, None, :]
#     sinusoid_inp = sinusoid_inp[..., None, :]

#     sinusoid_inp = rearrange(sinusoid_inp, "... seq head dim -> ... head seq dim")
#     sin = torch.sin(sinusoid_inp)
#     cos = torch.cos(sinusoid_inp)

#     return sin.bfloat16().cuda(), cos.bfloat16().cuda()


# def apply_rope(inputs: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#     sin, cos = rope_cache

#     first_half, second_half = torch.chunk(inputs, 2, -1)
#     first_part = first_half * cos - second_half * sin
#     second_part = second_half * cos + first_half * sin
#     out = torch.concatenate([first_part, second_part], -1)
#     return out

def apply_rope(inputs: Tensor, positions: Tensor, max_wavelength: int = _MAX_WAVELENGTH) -> Tensor:
    dtype = inputs.dtype
    device = inputs.device
    head_dim = inputs.shape[-1]

    fraction = 2 * torch.arange(0, head_dim // 2, dtype=dtype, device=device) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., None] / timescale[None, None, :]
    sinusoid_inp = sinusoid_inp[..., None, :]

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    first_half, second_half = torch.chunk(inputs, 2, -1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = torch.concatenate([first_part, second_part], -1)

    return out
