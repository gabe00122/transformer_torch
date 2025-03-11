import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate, generalized_advantage_estimate

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
        self.linear2 = nn.Linear(d_model, d_model)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        
        # self.in_linear = nn.Linear(d_model, d_model)
        # self.activation = nn.SiLU()
        self.out_linear = nn.Linear(d_model, self.action_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
        # x = self.in_linear(x)
        # x = self.activation(x)
        x = self.out_linear(x)
        logits = x
        return Categorical(logits=logits)


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.in_linear = nn.Linear(d_model, d_model)
        self.activation = nn.LeakyReLU()
        self.out_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_linear(x)
        x = self.activation(x)
        x = self.out_linear(x)
        return x



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

i = 0

def loss_fn(model: RLTransformerModel, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
    discount = 0.92
    actor_coef = 2.0 #0.5
    entropy_coef = 0.003 #0.0001
    seq_length = observations.shape[1]
    
    positions = torch.arange(seq_length, device=observations.device)

    # Observation dimensions: (batch_size, context_size, ...)
    policy, values = torch.compile(model, mode="max-autotune")(observations, positions)
    policy.logits = policy.logits[:, :-1]

    next_values = values[:, 1:]
    values = values[:, :-1]

    next_values = next_values.detach()
    # advantage = target - values
    advantage, target = vec_generalized_advantage_estimate(discount, 0.89, values, next_values, rewards, terminated, time_dim=1)

    # Critic loss
    critic_loss = F.smooth_l1_loss(values, target) # torch.square(target - values)

    # Actor loss
    action_probability = policy.log_prob(actions)
    actor_loss = -(action_probability * advantage.detach())

    # Entropy regularization
    entropy = policy.entropy()
    entropy_loss = -entropy

    total_loss = (
        0.5 * critic_loss
        + actor_loss
        + entropy_coef * entropy_loss
    ).mean()

    return total_loss, actor_loss.mean(), critic_loss.mean()

import gymnasium as gym

@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
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
        
        # observation gets plus one because we need to store the next trailing observation
        self.obs_rollout = torch.zeros((self.batch_size, trajectory_length, obs_dim), device=device, dtype=torch.float32)
        self.action_rollout = torch.zeros((self.batch_size, trajectory_length - 1), device=device, dtype=torch.int64)
        self.reward_rollout = torch.zeros((self.batch_size, trajectory_length - 1), device=device, dtype=torch.float32)
        self.terminated_rollout = torch.zeros((self.batch_size, trajectory_length - 1), device=device, dtype=torch.bool)
        
        # Track cumulative rewards per environment
        self.cumulative_rewards = torch.zeros(self.batch_size, device=device, dtype=torch.float32)
        self.episode_lengths = torch.zeros(self.batch_size, device=device, dtype=torch.int64)
        self.completed_episodes = 0
        self.completed_rewards = []

    def rollout(self):
        self.model.eval()
        self.model.clear_kv_cache()

        obs, info = self.env.reset()
        obs_tensor = torch.from_numpy(obs).to(self.device)

        self.positions.zero_()
        self.obs_rollout[self.batch_idx, self.positions] = obs_tensor

        # Reset cumulative rewards for new episodes
        self.cumulative_rewards.zero_()
        self.episode_lengths.zero_()

        self.obs_rollout.zero_()
        self.action_rollout.zero_()
        self.reward_rollout.zero_()
        self.terminated_rollout.zero_()

        reward_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        terminated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        truncated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)

        for i in range(self.trajectory_length - 1):
            with torch.no_grad():
                action = sample_action(self.model, obs_tensor, self.positions)
            obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
            
            obs_tensor.copy_(torch.from_numpy(obs))
            reward_tensor.copy_(torch.from_numpy(reward))
            terminated_tensor.copy_(torch.from_numpy(terminated))
            truncated_tensor.copy_(torch.from_numpy(truncated))

            self.action_rollout[self.batch_idx, self.positions] = action
            self.reward_rollout[self.batch_idx, self.positions] = reward_tensor
            self.terminated_rollout[self.batch_idx, self.positions] = terminated_tensor | truncated_tensor

            # if i < self.trajectory_length - 1:
            self.positions += 1
            self.obs_rollout[self.batch_idx, self.positions] = obs_tensor

            # Update cumulative rewards
            self.cumulative_rewards += reward_tensor
            self.episode_lengths += 1

            # Track completed episodes
            done_mask = terminated_tensor | truncated_tensor
            if done_mask.any():
                for env_idx in torch.where(done_mask)[0]:
                    self.completed_episodes += 1
                    self.completed_rewards.append(self.cumulative_rewards[env_idx].item())
                    # Reset rewards and lengths for done environments
                    self.cumulative_rewards[env_idx] = 0
                    self.episode_lengths[env_idx] = 0

        # Calculate mean reward metrics
        mean_reward = 0.0
        if self.completed_rewards:
            mean_reward = sum(self.completed_rewards) / len(self.completed_rewards)
            # Keep only the most recent 100 episodes
            if len(self.completed_rewards) > 100:
                self.completed_rewards = self.completed_rewards[-100:]

        # breakpoint()
        self.model.train()
        return loss_fn(self.model, self.obs_rollout, self.action_rollout, self.reward_rollout, self.terminated_rollout), mean_reward



def train():
    torch.set_float32_matmul_precision('high')
    console = Console()

    batch_size = 8
    trajectory_length = 512
    device = torch.device("cuda")

    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    model = RLTransformerModel(
        action_dim=env.single_action_space.n,
        num_layers=12 // 4,
        num_heads=12 // 4,
        d_model=768 // 4,
        ffn_size=2048 // 4,
        activation=nn.LeakyReLU(),
        glu=True,
    )
    model.apply(init_weights)
    model.policy_head.out_linear.weight.data /= 100.0

    model.create_kv_cache(batch_size, trajectory_length, device=device)
    model.create_block_mask(trajectory_length)

    console.print(f"Model size: {abbreviate_number(get_param_count(model))} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, fused=True)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, T_max=1000, eta_min=1e-6)
    model.to(device)

    model(torch.zeros((batch_size, trajectory_length, 4), device=device), torch.zeros((batch_size, trajectory_length), device=device))

    trainer = Trainer(model, env, trajectory_length, device)

    # Track cumulative rewards for monitoring
    best_mean_reward = 0.0

    for epoch in track(range(1000), console=console, disable=True):
        optimizer.zero_grad()
        (loss, actor_loss, critic_loss), mean_reward = trainer.rollout()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        # Update best mean reward
        best_mean_reward = max(best_mean_reward, mean_reward)
        
        # Log metrics
        console.print(f"[Step {epoch}] Loss: {loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Mean Reward: {mean_reward:.2f} | Best Mean Reward: {best_mean_reward:.2f} | Episodes: {trainer.completed_episodes}")
        
if __name__ == "__main__":
    train()
