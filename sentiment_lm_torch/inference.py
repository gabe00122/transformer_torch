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
from sentiment_lm_torch.tokenizer import Tokenizer

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def sample_token(model, tokens, positions):
    with torch.no_grad():
        logits = model(tokens, positions, None)
        
        logits = logits.squeeze(1)
        logits /= 0.8

        dist = Categorical(logits=logits)
        token = dist.sample()
    return token


def inference_cli(temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9):
    torch.set_float32_matmul_precision('high')

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

    while True:
        prompt = Prompt.ask("Prompt", console=console)
        console.print(prompt, end="")

        if prompt == "exit":
            break
        tokens_list, initial_length = tokenizer.encode(prompt)
        
        model.init_kv_cache(1, context_size, device, torch.bfloat16)

        epochs = 1
        start_time = time.time()
        for _ in range(epochs):
            tokens = torch.tensor([[tokens_list[0]]], dtype=torch.int64, device=device)
            positions = torch.tensor([[0]], dtype=torch.int64, device=device)

            for i in range(initial_length, context_size):
                token = sample_token(model, tokens, positions)
                
                positions += 1
                tokens = token.unsqueeze(0).clone()
                
                token_item = token.item()
                if token_item == END_TOKEN:
                    break

                console.print(tokenizer.decode_token(int(token_item)), end="")
            console.print("\n")
        total_time = time.time() - start_time


if __name__ == "__main__":
    inference_cli()
