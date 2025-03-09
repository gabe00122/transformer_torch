import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask

from rich.console import Console
from rich.progress import track

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

    def forward(self, inputs: torch.Tensor, positions: Tensor, *, policy_only: bool = False) -> tuple[Distribution, torch.Tensor]:
        block_mask: BlockMask | None = None
        if self.training:
            block_mask = self.block_mask
        
        x = self.observation_encoder(inputs)

        for layer in self.layers:
            x = layer(x, positions, block_mask)

        x = self.output_norm(x)

        policy = self.policy_head(x)
        if policy_only:
            return policy

        value = self.value_head(x)
        value = value.squeeze(-1)

        return policy, value

# observation: Observation
# action: Action
# reward: chex.Array
# next_observation: Observation
# terminated: Done
# truncated: Done

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=False)
def loss_fn(model: RLTransformerModel, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
    discount = 0.99
    actor_coef = 1.0
    entropy_coef = 0.01
    seq_length = observations.shape[1]
    
    positions = torch.arange(seq_length, device=observations.device)

    # Observation dimensions: (batch_size, context_size, ...)
    policy, values = model(observations, positions)

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
    entropy = policy.entropy().mean()
    entropy_loss = -entropy

    total_loss = (
        critic_loss
        + actor_coef * actor_loss
        + entropy_coef * entropy_loss
    ).mean()

    return total_loss

import gymnasium as gym

@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=False)
def sample_action(model: RLTransformerModel, obs: Tensor, positions: Tensor) -> Tensor:
    policy = model(obs[:, None, ...], positions, policy_only=True)
    action: Tensor = policy.sample()
    action = action.squeeze(-1)
    return action

class Trainer:
    def __init__(self, model: RLTransformerModel, env: gym.vector.VectorEnv, trajectory_length: int, device: torch.device = torch.device("cuda")):
        self.model = model
        self.env = env
        self.trajectory_length = trajectory_length
        self.batch_size = env.num_envs
        self.device = device

        obs_dim = env.single_observation_space.shape[0]

        self.batch_idx = torch.arange(self.batch_size, device=device, dtype=torch.int64)
        self.positions = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.int64)
        
        self.obs_rollout = torch.zeros((self.batch_size, trajectory_length, obs_dim), device=device, dtype=torch.float32)
        self.action_rollout = torch.zeros((self.batch_size, trajectory_length), device=device, dtype=torch.int64)
        self.reward_rollout = torch.zeros((self.batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.terminated_rollout = torch.zeros((self.batch_size, trajectory_length), device=device, dtype=torch.bool)

    def rollout(self):
        self.model.eval()
        self.model.clear_kv_cache()

        obs, info = self.env.reset()
        torch_obs = torch.from_numpy(obs).to(self.device)

        self.positions.zero_()
        self.obs_rollout[self.batch_idx, self.positions] = torch_obs

        # numpy_action = np.zeros(self.batch_size, dtype=np.int32)
        reward_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        terminated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)


        # done = False
        for i in range(self.trajectory_length):
            with torch.no_grad():
                action = sample_action(self.model, torch_obs, self.positions)
            obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
            torch_obs = torch.from_numpy(obs).to(self.device)

            reward_tensor.copy_(torch.from_numpy(reward))
            terminated_tensor.copy_(torch.from_numpy(terminated))

            self.action_rollout[self.batch_idx, self.positions] = action
            self.reward_rollout[self.batch_idx, self.positions] = reward_tensor
            self.terminated_rollout[self.batch_idx, self.positions] = terminated_tensor

            if i < self.trajectory_length - 1:
                self.positions += 1
                self.obs_rollout[self.batch_idx, self.positions] = torch_obs

            # done = terminated[0] or truncated[0]


        self.model.train()
        return loss_fn(self.model, self.obs_rollout, self.action_rollout, self.reward_rollout, self.terminated_rollout)



def train():
    torch.set_float32_matmul_precision('high')
    console = Console()

    batch_size = 64
    trajectory_length = 512
    device = torch.device("cuda")

    env = gym.make_vec("CartPole-v1", num_envs=batch_size, vectorization_mode="sync")

    model = RLTransformerModel(
        action_dim=env.single_action_space.n,
        num_layers=2,
        num_heads=4,
        d_model=128,
        ffn_size=128,
        activation=nn.SiLU(),
        glu=False,
    )
    model.apply(init_weights)
    model.create_kv_cache(batch_size, trajectory_length, device=device)
    model.create_block_mask(trajectory_length)

    console.print(f"Model size: {abbreviate_number(get_param_count(model))} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    model(torch.zeros((batch_size, trajectory_length, 4), device=device), torch.zeros((batch_size, trajectory_length), device=device))

    trainer = Trainer(model, env, trajectory_length, device)

    for _ in track(range(100), console=console):
        model.zero_grad()
        loss = trainer.rollout()
        loss.backward()
        optimizer.step()
        console.print(f"Loss: {loss.item():.4f}")
        
if __name__ == "__main__":
    train()
