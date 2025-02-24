import time
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.attention.flex_attention import BlockMask

from rich.console import Console
from rich.prompt import Prompt

from sentiment_lm_torch.constants import END_TOKEN, SPECIAL_TOKENS
from sentiment_lm_torch.model.attention import KVCache, inference_block_mask, create_kv_cache
from sentiment_lm_torch.model.positional_embeddings import create_rope_cache
from sentiment_lm_torch.tokenizer import Tokenizer

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def sample_token(model, tokens, rope_cache: tuple[Tensor, Tensor], block_mask: BlockMask, kv_cache: KVCache):
    with torch.no_grad():
        logits, kv_cache = model(tokens, rope_cache, block_mask, kv_cache)
        
        logits = logits.squeeze(1)
        # Use gather to select the logits we want
        logits /= 0.8

        dist = Categorical(logits=logits)
        token = dist.sample()
    return token, kv_cache


def inference_cli(temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9):
    # torch.set_float32_matmul_precision('high')
    # torch._dynamo.config.capture_scalar_outputs = True

    console = Console()
    device = torch.device("cuda")

    cfg = OmegaConf.load("./config.yaml")
    context_size = cfg.dataset.context_size
    vocab_size = cfg.vocab.size + SPECIAL_TOKENS

    tokenizer = Tokenizer(cfg.vocab.file, context_size)

    model: nn.Module = instantiate(cfg.model, vocab_size=vocab_size)
    
    sd = torch.load("./model.pth")
    model.load_state_dict(sd)
    model = model.to(torch.bfloat16)
    model = model.to(device)

    # rope_cache = create_rope_cache(cfg.model.d_model // cfg.model.num_heads, torch.tensor(0))
    # block_mask = inference_block_mask(context_size, 1)

    while True:
        kv_cache = create_kv_cache(cfg.model.num_layers, 1, cfg.model.d_model, cfg.model.num_heads, context_size, device)

        prompt = Prompt.ask("Prompt", console=console)
        console.print(prompt, end="")

        if prompt == "exit":
            break
        tokens_list, initial_length = tokenizer.encode(prompt)
        
        # console.print("Warmup")
        # tokens = torch.tensor(tokens_list, dtype=torch.int32).to("cuda")
        # index = torch.tensor(initial_length, dtype=torch.int32).to("cuda")
        # sample_token(model, tokens, rope_cache, block_mask, index)
        rope_cache: tuple[Tensor, Tensor] = create_rope_cache(cfg.model.d_model // cfg.model.num_heads, torch.arange(context_size))
        block_mask = None

        epochs = 1
        start_time = time.time()
        for _ in range(epochs):
            tokens = torch.tensor([[tokens_list[0]]], dtype=torch.int64, device=device)
            # index = torch.tensor(initial_length - 1, dtype=torch.int64).to("cuda")

            for _ in range(initial_length, context_size):
                # torch.compiler.cudagraph_mark_step_begin()
                token, kv_cache = sample_token(model, tokens, rope_cache, block_mask, kv_cache)
                kv_cache = KVCache(
                    key=kv_cache.key.clone(),
                    value=kv_cache.value.clone(),
                    length=kv_cache.length.clone() + 1,
                )
                # tokens[index] = token
                
                tokens[0] = token
                token_item = token.item()
                if token_item == END_TOKEN:
                    break

                console.print(tokenizer.decode_token(int(token_item)), end="")
            console.print("\n")
        total_time = time.time() - start_time
        # total_tokens = index.item() #epochs * context_size
        # console.print(f"Tokens per second: {total_tokens / total_time:.2f}")

        # console.print(f"Time: {total_time:.2f}s")


if __name__ == "__main__":
    inference_cli()
