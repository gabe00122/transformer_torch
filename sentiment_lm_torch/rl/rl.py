from turtle import done
import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask

from rich.console import Console

from sentiment_lm_torch.model.transformer import TransformerLayer
from sentiment_lm_torch.model.util import init_weights
from sentiment_lm_torch.utils import get_param_count, abbreviate_number
from sentiment_lm_torch.model.attention import causal_block_mask

class ObservationEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.linear = nn.Linear(d_model, action_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
        logits = self.linear(x) / 100.0
        return Categorical(logits=logits)


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)



class RLTransformerModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        *,
        activation: nn.Module = nn.SiLU(),
        glu: bool = True,
        context_size: int,
        dtype: torch.dtype=torch.float32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.ffn_size = ffn_size
        
        self.activation = activation
        self.glu = glu
        self.context_size = context_size
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

        self.output_norm = nn.RMSNorm(d_model, dtype=dtype)

        self.observation_encoder = ObservationEncoder(d_model)
        self.policy_head = PolicyHead(d_model, action_dim)
        self.value_head = ValueHead(d_model)

    def create_kv_cache(self, batch_size: int, context_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        for layer in self.layers:
            layer.attention.init_kv_cache(batch_size, context_size, device, dtype)

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.clear_kv_cache()
    
    def create_block_mask(self, seq_length: int):
        self.block_mask = causal_block_mask(seq_length)

    def forward(self, inputs: torch.Tensor, positions: Tensor) -> tuple[Distribution, torch.Tensor]:
        block_mask: BlockMask | None = None
        if self.training:
            block_mask = self.block_mask
        
        x = self.observation_encoder(inputs)

        for layer in self.layers:
            x = layer(x, positions, block_mask)

        x = self.output_norm(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        value = value.squeeze(-1)

        return policy, value

# observation: Observation
# action: Action
# reward: chex.Array
# next_observation: Observation
# terminated: Done
# truncated: Done

def loss_fn(model: RLTransformerModel, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
    discount = 0.99
    actor_coef = 1.0
    entropy_coef = 0.01
    seq_length = observations.shape[1]
    
    # Compute the policy loss
    positions = torch.arange(seq_length, device=observations.device)

    # Observation dimensions: (batch_size, context_size, ...)

    policy, values = model(observations, positions)

    breakpoint()
    next_values = torch.roll(values, shifts=-1, dims=-1)
    next_values[:, -1] = 0.0


    next_values = next_values.detach()

    target = rewards + discount * next_values * ~terminated
    
    temporal_difference_error = target - values

    # Critic loss
    critic_loss = temporal_difference_error ** 2

    # Actor loss
    action_probability = policy.log_prob(actions)
    actor_loss = -(action_probability * temporal_difference_error.detach())

    # Entropy regularization
    # entropy_seed = rngs.entropy()
    # entropy_loss = -policy.entropy(seed=entropy_seed)

    entropy = policy.entropy().mean()
    entropy_loss = -entropy

    total_loss = (
        critic_loss
        + actor_coef * actor_loss
        + entropy_coef * entropy_loss
    ).mean()

    return total_loss

import gymnasium as gym


def sample_action(model: RLTransformerModel, obs: Tensor, positions: Tensor) -> Tensor:
    with torch.no_grad():
        policy, _ = model(obs[:, None, ...], positions)
        action: Tensor = policy.sample()
        action = action.squeeze(-1)
        return action

def rollout(model: RLTransformerModel, env: gym.vector.VectorEnv, trajectory_length: int):
    batch_size = env.num_envs
    device = torch.device("cuda")
    # breakpoint()

    obs_rollout = torch.zeros((batch_size, trajectory_length, 4), device=device, dtype=torch.float32)
    action_rollout = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.int64)
    reward_rollout = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
    terminated_rollout = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.bool)

    model.eval()

    obs, info = env.reset()
    torch_obs = torch.from_numpy(obs).to(device)
    positions = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)

    batch_idx = torch.arange(batch_size, device=device, dtype=torch.int64)
    obs_rollout[batch_idx, positions] = torch_obs

    # done = False
    for i in range(trajectory_length):
        action = sample_action(model, torch_obs, positions)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        torch_obs = torch.from_numpy(obs).to(device)

        action_rollout[batch_idx, positions] = action
        reward_rollout[batch_idx, positions] = torch.from_numpy(reward).to(device)
        terminated_rollout[batch_idx, positions] = torch.from_numpy(terminated).to(device)

        if i < trajectory_length - 1:
            positions += 1
            obs_rollout[batch_idx, positions] = torch_obs

        # done = terminated[0] or truncated[0]


    model.train()
    loss = loss_fn(model, obs_rollout, action_rollout, reward_rollout, terminated_rollout)
    print(loss)



def train():
    console = Console()

    batch_size = 2
    trajectory_length = 512
    device = torch.device("cuda")

    env = gym.make_vec("CartPole-v1", num_envs=batch_size, render_mode="human")

    model = RLTransformerModel(
        action_dim=env.single_action_space.n,
        num_layers=2,
        num_heads=4,
        d_model=128,
        ffn_size=128,
        context_size=500,
        activation=nn.SiLU(),
        glu=False,
    )
    model.apply(init_weights)
    model.create_kv_cache(batch_size, trajectory_length, device=device)
    model.create_block_mask(trajectory_length)

    model.to(device)

    rollout(model, env, trajectory_length)
        
if __name__ == "__main__":
    train()
