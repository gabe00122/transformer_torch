import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from rich.progress import track
from rich.console import Console
from more_itertools import ncycles
import wandb

from sentiment_lm_torch.constants import SPECIAL_TOKENS
from sentiment_lm_torch.dataset import SentimentDataset
from sentiment_lm_torch.model.util import init_weights
from sentiment_lm_torch.scheduler import get_cosine_schedule_with_warmup
from sentiment_lm_torch.utils import get_param_count, abbreviate_number
from sentiment_lm_torch.constants import EMPTY_TOKEN


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    console = Console()

    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.batch_size // cfg.accumulation_steps

    training_dataloader = DataLoader(
        SentimentDataset(cfg.dataset.training_path),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    context_size = cfg.dataset.context_size
    vocab_size = cfg.vocab.size + SPECIAL_TOKENS
    total_steps = len(training_dataloader) * cfg.epochs
    total_gradient_steps = total_steps // cfg.accumulation_steps
    total_tokens = len(training_dataloader) * batch_size * context_size

    model: nn.Module = instantiate(cfg.model, vocab_size=vocab_size, context_size=context_size)
    model.apply(init_weights)

    optimizer = instantiate(cfg.optimizer,model.parameters(), fused=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.scheduler.warmup_steps, total_gradient_steps)

    console.print(f"Token count: {abbreviate_number(total_tokens)}")
    console.print(f"Parameter count: {abbreviate_number(get_param_count(model))}")

    model = model.to(device)    
    train_step_compiled = torch.compile(train_step, fullgraph=True)

    wandb.init(project="sentiment_lm_torch")

    loss_metric = 0

    model.train()
    for step, tokens in track(enumerate(ncycles(training_dataloader, cfg.epochs)), total=total_steps, console=console):
        tokens = tokens.to(device)
        
        loss = train_step_compiled(model, tokens, cfg.accumulation_steps)
        loss.backward()

        loss_metric += loss.item()

        if step % cfg.accumulation_steps == cfg.accumulation_steps - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            console.print(f"Step {step}, Loss: {loss_metric}")
            wandb.log({"loss": loss_metric}, step=step)
            loss_metric = 0

    torch.save(model.state_dict(), "checkpoints/model.pth")


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1, logits.size(-1))
    targets = labels.long().view(-1)

    return F.cross_entropy(logits, targets, ignore_index=EMPTY_TOKEN)


def train_step(model, tokens, accumulation_steps):
    input_tokens = tokens[:, :-1]
    labels = tokens[:, 1:]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(input_tokens)
        return loss_fn(logits, labels) / accumulation_steps


if __name__ == "__main__":
    train()
