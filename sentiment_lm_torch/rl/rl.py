import torch
from torch import nn
from torch.distributions import Categorical, Distribution

from sentiment_lm_torch.model.transformer import TransformerLayer
from sentiment_lm_torch.model.util import init_weights

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
        return Categorical(logits=self.linear(x))


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

        attention_mask = torch.tril(torch.ones((self.context_size, self.context_size), dtype=torch.bool))
        self.register_buffer("attention_mask", attention_mask)

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

    def forward(self, inputs: torch.Tensor, segment_positions: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        x = self.observation_encoder(inputs)

        for layer in self.layers:
            x = layer(x, segment_positions, self.attention_mask)

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
    
    # Compute the policy loss
    segment_positions = torch.arange(observations.shape[1], device=observations.device)

    # Observation dimensions: (batch_size, context_size, ...)

    policy, values = model(observations, segment_positions)

    next_values = torch.cat([values[:, 1:], torch.zeros((values.size(0), 1), device=values.device)], dim=1)

    next_values = next_values.detach()

    target = rewards + discount * next_values * (1.0 - terminated)
    
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


def sample_action(model: RLTransformerModel, obs_buffer: torch.Tensor, time_step_index: torch.Tensor) -> torch.Tensor:
    # time_step_index (batchsize,)

    segment_positions = torch.arange(obs_buffer.shape[1], device=obs_buffer.device)
    policy, _ = model(obs_buffer, segment_positions)
    action = policy.sample()
    action = action[:, time_step_index]
    return action


def train():
    batch_size = 1

    env = gym.make_vec("CartPole-v1", num_envs=batch_size, render_mode="human")

    model = RLTransformerModel(
        action_dim=env.single_action_space.n,
        num_layers=2,
        num_heads=2,
        d_model=100,
        ffn_size=100,
        context_size=500,
        activation=nn.SiLU(),
        glu=False,
    )
    model.apply(init_weights)

    trajectory_length = 500
    observation_dim = 4

    time_step_index = torch.zeros((batch_size,), dtype=torch.int)
    obs_buffer = torch.zeros((batch_size, trajectory_length, observation_dim), dtype=torch.float32)

    obs, info = env.reset()
    
    obs_buffer[:, 0] = torch.from_numpy(obs).float()
    actions = sample_action(model, obs_buffer, time_step_index)

    for _ in range(1, trajectory_length):
        obs, rewards, terminated, truncated, info = env.step(actions.numpy())

        obs_buffer[:, time_step_index + 1] = torch.from_numpy(obs).float()
        time_step_index += 1

        if terminated or truncated:
            break
        
if __name__ == "__main__":
    train()
