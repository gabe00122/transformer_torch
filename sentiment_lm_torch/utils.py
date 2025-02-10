import torch

def get_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def abbreviate_number(n: int) -> str:
    """Abbreviates a number to a string with suffixes for thousands, millions, etc."""
    if n < 1e3:
        return str(n)
    elif n < 1e6:
        return f"{n / 1e3:.1f}K"
    elif n < 1e9:
        return f"{n / 1e6:.1f}M"
    elif n < 1e12:
        return f"{n / 1e9:.1f}B"
    else:
        return f"{n / 1e12:.1f}T"
