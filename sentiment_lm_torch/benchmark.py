import time
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.attention.flex_attention import BlockMask

from rich.console import Console
from rich.prompt import Prompt
from rich.progress import track

from sentiment_lm_torch.constants import END_TOKEN, SPECIAL_TOKENS
from sentiment_lm_torch.model.attention import causal_block_mask
from sentiment_lm_torch.model.positional_embeddings import create_rope_cache
from sentiment_lm_torch.tokenizer import Tokenizer

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def sample_token(model, tokens, rope_cache: tuple[Tensor, Tensor], block_mask: BlockMask, indices: Tensor):
    with torch.no_grad():
        logits = model(tokens, rope_cache, block_mask)
        
        # Use gather to select the logits we want
        batch_idx = torch.arange(tokens.size(0), device=tokens.device, dtype=torch.int64)
        logits = logits[batch_idx, indices]
        logits /= 0.8

        dist = Categorical(logits=logits)
        token = dist.sample()
    return token


def benchmark():
    console = Console()

    cfg = OmegaConf.load("./config.yaml")
    context_size = cfg.dataset.context_size
    vocab_size = cfg.vocab.size + SPECIAL_TOKENS

    warmup = 10
    iterations = 100

    batch = 32

    model: nn.Module = instantiate(cfg.model, vocab_size=vocab_size)
    
    sd = torch.load("./model.pth")
    model.load_state_dict(sd)
    model = model.to(torch.bfloat16)
    model = model.to("cuda")

    rope_cache = create_rope_cache(cfg.model.d_model // cfg.model.num_heads, torch.arange(context_size))
    block_mask = causal_block_mask(context_size)

    def benchmark_nocache():
        tokens = torch.zeros((batch, context_size), dtype=torch.int64, device="cuda")
        index = torch.zeros(batch, dtype=torch.int64, device="cuda")

        for _ in range(context_size - 1):
            token = sample_token(model, tokens, rope_cache, block_mask, index)
            index += 1

            batch_idx = torch.arange(batch, device=tokens.device, dtype=torch.int64)
            tokens[batch_idx, index] = token

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
    console.print(f"Tokens per second: {batch * context_size * iterations / delta_time}")


if __name__ == "__main__":
    benchmark()
