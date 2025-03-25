import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
import numpy as np
import gymnasium as gym
from mettagrid.gym_wrapper import make, MultiToDiscreteWrapper
from typing import TypedDict, Optional, List, Tuple, Union

from einops import rearrange

from rich.console import Console
from rich.progress import track

from sentiment_lm_torch.model.transformer import TransformerLayer
from sentiment_lm_torch.model.util import init_weights
from sentiment_lm_torch.rl.metta_utils import ObservationNormalizer
from sentiment_lm_torch.utils import get_param_count, abbreviate_number
from sentiment_lm_torch.model.attention import causal_block_mask

class RLConfig(TypedDict):
    # Environment config
    env_name: str
    num_agents: int
    max_steps: int
    
    # Training config
    total_gradient_steps: int
    minibatch_steps: int
    trajectory_length: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    device: str
    
    # Model architecture config
    d_model: int
    num_layers: int
    num_heads: int
    ffn_size: int
    activation: str  # 'relu', 'leaky_relu', 'silu'
    use_glu: bool
    
    # CNN encoder config
    cnn_channels: List[int]
    
    # Loss function config
    vf_coef: float
    entropy_coef: float
    vf_clip: float
    gamma: float
    gae_lambda: float
    
    # Optimizer config
    optimizer: str  # 'adam', 'adamw'
    betas: Tuple[float, float]
    eps: float
    
    # Optional scheduler config
    use_scheduler: bool
    scheduler_type: Optional[str]  # 'linear', 'cosine', etc.
    scheduler_start_factor: float
    scheduler_end_factor: float

class MlpObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = F.leaky_relu(x)

        return x

def _convolution_shape(shape, kernel_size, stride):
    return tuple((x - kernel_size) // stride + 1 for x in shape)

class MetaCnnEncoder(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, ...],
        d_model: int,
        activation: nn.Module,
        channels=[32, 64],
        *,
        grid_features: list[str] = [],
    ):
        super().__init__()
        C = img_shape[-1]
        img_shape = img_shape[:-1]

        self.conv1 = nn.Conv2d(C, channels[0], kernel_size=5, stride=3)
        img_shape = _convolution_shape(img_shape, 5, 3)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1)
        img_shape = _convolution_shape(img_shape, 3, 1)
        
        self.linear = nn.Linear(img_shape[0] * img_shape[1] * channels[1], d_model)
        self.activation = activation

        self.object_normalizer = ObservationNormalizer(grid_features)

    def forward(self, obs: torch.Tensor):
        b, l, _, _, _ = obs.shape
        obs = rearrange(obs, "b l h w c -> (b l) c h w")

        if self.object_normalizer is not None:
            obs = self.object_normalizer(obs)
        
        x = self.activation(self.conv1(obs))
        x = self.activation(self.conv2(x))
        x = rearrange(x, "(b l) c h w -> b l (h w c)", b=b, l=l)
        x = self.linear(x)

        return x


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.p_linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, self.action_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
        x = self.p_linear(x)
        x = F.leaky_relu(x)
        x = self.linear(x)
        
        return Categorical(logits=x)


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.p_linear = nn.Linear(d_model, d_model)
        self.activation = nn.LeakyReLU()
        self.out_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p_linear(x)
        x = self.activation(x)
        x = self.out_linear(x)
        return x


class RLTransformerModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        observation_encoder: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        *,
        activation: nn.Module = nn.SiLU(),
        glu: bool = True,
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.ffn_size = ffn_size
        
        self.activation = activation
        self.glu = glu
        self.dtype = dtype

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerLayer(
                    num_heads,
                    d_model,
                    ffn_size,
                    activation=self.activation,
                    glu=glu,
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.output_norm = nn.LayerNorm(d_model, dtype=dtype)

        self.observation_encoder = observation_encoder
        self.policy_head = policy_head
        self.value_head = value_head

    def create_kv_cache(self, batch_size: int, context_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        for layer in self.layers:
            layer.attention.init_kv_cache(batch_size, context_size, device, dtype)

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.clear_kv_cache()

    def forward(self, inputs: torch.Tensor, positions: Tensor, *, block_mask: BlockMask | None = None) -> tuple[Distribution, torch.Tensor]:
        x = self.observation_encoder(inputs)

        for layer in self.layers:
            x = layer(x, positions, block_mask)

        x = self.output_norm(x)

        policy = self.policy_head(x)

        value = self.value_head(x)
        value = value.squeeze(-1)

        return policy, value


@torch.compile(mode="max-autotune", disable=False)
def loss_fn(model: RLTransformerModel, rollout: 'Rollout', block_mask: BlockMask, config: RLConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vf_coef = config["vf_coef"]
    entropy_coef = config["entropy_coef"]
    vf_clip = config["vf_clip"]

    obs = rollout.obs
    advantage = rollout.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    positions = torch.arange(obs.size(1), device=torch.device("cuda"), dtype=torch.int64)[None, :]

    # Observation dimensions: (batch_size, context_size, ...)
    policy, values = model(obs, positions, block_mask=block_mask)
    log_probs = policy.log_prob(rollout.actions)

    value_losses = torch.square(values - rollout.target)
    value_loss = 0.5 * value_losses.mean()

    ratio = torch.exp(log_probs - rollout.log_prob)

    loss_actor1 = ratio * advantage
    loss_actor2 = torch.clamp(ratio, 1.0 - vf_clip, 1.0 + vf_clip) * advantage

    actor_loss = -torch.min(loss_actor1, loss_actor2).mean()

    # Entropy regularization
    entropy = policy.entropy()
    entropy_loss = -entropy.mean()

    total_loss = vf_coef * value_loss + actor_loss + entropy_coef * entropy_loss

    return total_loss, actor_loss, value_loss

@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=False)
def sample_action(model: RLTransformerModel, obs: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    policy, value = model(obs[:, None, ...], positions)
    action: Tensor = policy.sample()
    log_prob = policy.log_prob(action)

    action = action.squeeze(-1)
    log_prob = log_prob.squeeze(-1)
    value = value.squeeze(-1)
    return action, log_prob, value

class Rollout:
    def __init__(self, batch_size: int, trajectory_length: int, obs_dims: tuple[int, ...], device: torch.device):
        # observation gets plus one because we need to store the next trailing observation
        self.obs = torch.zeros((batch_size, trajectory_length, *obs_dims), device=device, dtype=torch.float32)
        self.actions = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.int64)
        self.reward = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.terminated = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.bool)
        self.truncated = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.bool)

        self.log_prob = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.values = torch.zeros((batch_size, trajectory_length + 1), device=device, dtype=torch.float32)

        self.advantage = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.target = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
    
    def calculate_advantage(self, config: RLConfig):
        values = self.values[..., :-1]
        next_values = self.values[..., 1:]

        # Sets the last value to be terminated because the end of the trajectory is always out of distribution for the transformer.
        self.terminated[..., -1] = True

        self.advantage, self.target = vec_generalized_advantage_estimate(
            config["gamma"],
            config["gae_lambda"],
            values,
            next_values,
            self.reward,
            self.truncated | self.terminated,
            self.terminated,
            time_dim=1
        )


class Trainer:
    def __init__(self, model: RLTransformerModel, env: gym.vector.VectorEnv, trajectory_length: int, device: torch.device = torch.device("cuda")):
        self.model = model
        self.env = env
        self.trajectory_length = trajectory_length
        self.batch_size = env.unwrapped.num_agents
        self.device = device

        obs_dims = env.unwrapped.single_observation_space.shape

        self.batch_idx = torch.arange(self.batch_size, device=device, dtype=torch.int64)
        self.positions = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.int64)
        
        # observation gets plus one because we need to store the next trailing observation
        self.rollout = Rollout(self.batch_size, trajectory_length, obs_dims, device)
        
        # Track cumulative rewards per environment
        self.cumulative_rewards = torch.zeros(self.batch_size, device=device, dtype=torch.float32)
        self.episode_lengths = torch.zeros(self.batch_size, device=device, dtype=torch.int64)
        self.completed_episodes = 0
        self.completed_rewards = []

        self.block_mask = causal_block_mask(trajectory_length)

        self.reward_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        self.terminated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.truncated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.obs_tensor = torch.zeros((self.batch_size, *obs_dims), device=self.device, dtype=torch.float32)

    def create_rollout(self):
        # Reset cumulative rewards for new episodes
        self.cumulative_rewards.zero_()
        self.episode_lengths.zero_()
        self.positions.zero_()

        obs, _ = self.env.reset()
        self.obs_tensor.copy_(torch.from_numpy(obs), non_blocking=True)

        for i in range(self.trajectory_length):
            with torch.no_grad():
                action, log_prob, value = sample_action(self.model, self.obs_tensor, self.positions)
            np_action = action.cpu().numpy()
            obs, reward, terminated, truncated, _ = self.env.step(np_action)
            
            self.reward_tensor.copy_(torch.from_numpy(reward), non_blocking=True)
            self.terminated_tensor.copy_(torch.from_numpy(terminated), non_blocking=True)
            self.truncated_tensor.copy_(torch.from_numpy(truncated), non_blocking=True)

            self.rollout.obs[:, i] = self.obs_tensor
            self.rollout.actions[:, i] = action
            self.rollout.log_prob[:, i] = log_prob
            self.rollout.values[:, i] = value
            self.rollout.reward[:, i] = self.reward_tensor
            self.rollout.terminated[:, i] = self.terminated_tensor
            self.rollout.truncated[:, i] = self.truncated_tensor
            
            self.obs_tensor.copy_(torch.from_numpy(obs), non_blocking=True)
            self.positions += 1
        
        with torch.no_grad():
            _, _, value = sample_action(self.model, self.obs_tensor, self.positions)
        self.rollout.values[:, -1] = value

        self.rollout.calculate_advantage(config)
        return self.rollout, self.rollout.reward.sum()


def train(trial_id: int, config: RLConfig):
    torch.set_float32_matmul_precision('high')
    console = Console()

    env = MultiToDiscreteWrapper(make(config["env_name"], overrides=[
        f"game.num_agents={config['num_agents']}", 
        # f"game.max_steps={config['max_steps']}"
    ]))
    image_shape = env.unwrapped.single_observation_space.shape
    action_dim = env.action_space.n
    num_agents = env.unwrapped.num_agents

    print(image_shape)
    print(action_dim)
    print(num_agents)

    batch_size = num_agents
    device = torch.device(config["device"])

    # Create activation function based on config
    activation_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU()
    }
    activation = activation_map[config["activation"]]

    observation_encoder = MetaCnnEncoder(
        img_shape=image_shape,
        d_model=config["d_model"],
        activation=activation,
        channels=config["cnn_channels"],
        grid_features=env.unwrapped.grid_features
    )
    policy_head = PolicyHead(d_model=config["d_model"], action_dim=2)
    value_head = ValueHead(d_model=config["d_model"])

    model = RLTransformerModel(
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        ffn_size=config["ffn_size"],
        activation=activation,
        glu=config["use_glu"],
        observation_encoder=observation_encoder,
        policy_head=policy_head,
        value_head=value_head,
    )
    model.apply(init_weights)
    nn.init.orthogonal_(model.policy_head.linear.weight)
    model.policy_head.linear.weight.data *= 0.01
    nn.init.orthogonal_(model.value_head.out_linear.weight)

    model.create_kv_cache(batch_size, config["trajectory_length"] + 1, device=device)

    console.print(f"Model size: {abbreviate_number(get_param_count(model))} parameters")
    
    # Create optimizer based on config
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config["betas"],
            eps=config["eps"]
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config["betas"],
            eps=config["eps"]
        )

    # Create scheduler if configured
    scheduler = None
    if config["use_scheduler"]:
        if config["scheduler_type"] == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                total_iters=config["total_gradient_steps"],
                start_factor=config["scheduler_start_factor"],
                end_factor=config["scheduler_end_factor"]
            )

    model.to(device)

    trainer = Trainer(model, env, config["trajectory_length"], device)

    total_rollouts = config["total_gradient_steps"] // config["minibatch_steps"]
    objective = 0.0

    # Track cumulative rewards for monitoring
    best_mean_reward = 0.0

    for epoch in track(range(config["total_gradient_steps"]), console=console, disable=True):
        optimizer.zero_grad()

        if epoch % config["minibatch_steps"] == 0:
            model.eval()
            rollout, mean_reward = trainer.create_rollout()
            objective += mean_reward.item() / total_rollouts

        model.train()
        loss, actor_loss, critic_loss = loss_fn(model, rollout, trainer.block_mask, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Update best mean reward
        best_mean_reward = max(best_mean_reward, mean_reward)
        
        # Log metrics
        console.print(f"[Step {epoch}] Loss: {loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Mean Reward: {mean_reward:.4f} | Best Mean Reward: {best_mean_reward:.4f} | Episodes: {trainer.completed_episodes}")
    
    torch.save(model.state_dict(), f"trial_{trial_id}.pth")
    return objective


def enjoy():
    env = gym.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda")

    model = RLTransformerModel(
        action_dim=2,
        num_layers=2,
        num_heads=8,
        d_model=256,
        ffn_size=256,
        activation=nn.LeakyReLU(),
        glu=False,
    )
    model.load_state_dict(torch.load("pocp2.pth"))
    model.to(device)
    model.eval()

    model.create_kv_cache(1, 512, device=device)
    obs, info = env.reset()
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    positions = torch.zeros((1, 1), device=device, dtype=torch.int64)

    for _ in range(10000):
        with torch.no_grad():
            action, log_prob, value = sample_action(model, obs_tensor, positions)
        np_action = action.squeeze().cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(np_action)

        obs_tensor.copy_(torch.from_numpy(obs).unsqueeze(0))
        positions += 1

        if terminated or truncated:
            obs, info = env.reset()
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            positions = torch.zeros((1, 1), device=device, dtype=torch.int64)

import optuna

def objective(trial_id: int, trial: optuna.Trial):
    config: RLConfig = {
        "env_name": "bases",
        "num_agents": 28,
        "max_steps": 256,
        "total_gradient_steps": 10_000,
        "minibatch_steps": 3,
        "trajectory_length": 256,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.05, 1.0, step=0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 2.0),
        "entropy_coef": trial.suggest_float("entropy_coef", 0.0001, 0.1, log=True),
        "vf_clip": trial.suggest_float("vf_clip", 0.05, 0.5),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
        "betas": (
            trial.suggest_float("beta1", 0.8, 0.999), 
            trial.suggest_float("beta2", 0.9, 0.9999)
        ),
        "device": "cuda",
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 8,
        "ffn_size": 512,
        "activation": "silu",
        "use_glu": False,
        "cnn_channels": [32, 64],
        "optimizer": "adamw",
        "eps": 1e-12,
        "use_scheduler": False,
        "scheduler_type": None,
        "scheduler_start_factor": 1.0,
        "scheduler_end_factor": 0.0
    }

    outcome = train(trial_id, config)
    return outcome

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, group=True))
    study.optimize(lambda trial: objective(trial.number, trial), n_trials=200)
