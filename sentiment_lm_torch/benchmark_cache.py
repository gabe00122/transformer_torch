import time
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from rich.console import Console

from sentiment_lm_torch.constants import SPECIAL_TOKENS, START_TOKEN
from sentiment_lm_torch.model.transformer import TransformerModel


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def sample_token(model, tokens, positions):
    logits = model(tokens, positions, None)

    logits /= 0.8

    dist = Categorical(logits=logits)
    token = dist.sample()
    return token

def benchmark():
    torch.set_float32_matmul_precision('high')
    console = Console()
    
    device = torch.device("cuda")
    cfg = OmegaConf.load("./config.yaml")
    context_size = cfg.dataset.context_size
    vocab_size = cfg.vocab.size + SPECIAL_TOKENS

    warmup = 10
    iterations = 100

    batch = 64

    model: TransformerModel = instantiate(cfg.model, vocab_size=vocab_size)
    
    sd = torch.load("./model.pth")
    model.load_state_dict(sd)
    model = model.to(torch.bfloat16)
    model = model.to(device)
    
    model.init_kv_cache(batch, context_size, device, torch.bfloat16)

    def benchmark_nocache():
        tokens = torch.ones((batch, 1), dtype=torch.int64, device=device)
        positions = torch.zeros((batch, 1), dtype=torch.int64, device=device)

        # model.init_kv_cache(batch, context_size, device, torch.bfloat16)
        model.clear_kv_cache()

        for _ in range(context_size - 1):
            with torch.no_grad():
                token = sample_token(model, tokens, positions)
                positions += 1
                tokens = token.clone()

    console.print("Warmup")
    for _ in range(warmup):
        benchmark_nocache()

    console.print("Benchmark")
    start_time = time.time()
    for _ in range(iterations):
        benchmark_nocache()
    end_time = time.time()
    
    delta_time = end_time - start_time
    console.print(f"Time: {delta_time}")
    console.print(f"Tokens per second: {(batch * context_size * iterations) / delta_time}")


if __name__ == "__main__":
    benchmark()
