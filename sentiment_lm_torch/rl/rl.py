import IPython
import rich
import torch
from torch import log_, nn, Tensor
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
import gymnasium as gym

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
        x = self.linear(x)
        x = F.leaky_relu(x)

        return x


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.p_linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, self.action_dim)

    def forward(self, x: torch.Tensor) -> Distribution:
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
        x = F.leaky_relu(x)
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

        self.output_norm = nn.LayerNorm(d_model, dtype=dtype)

        self.observation_encoder = ObservationEncoder(d_model)
        self.policy_head = PolicyHead(d_model, action_dim)
        self.value_head = ValueHead(d_model)

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


@torch.compile(mode="max-autotune", disable=True)
def loss_fn(model: RLTransformerModel, rollout: 'Rollout', block_mask: BlockMask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vf_coef = 0.2
    entropy_coef = 0.002

    vf_clip = 0.2

    obs = rollout.obs
    rollout_values = rollout.values[:, :-1]

    positions = torch.arange(obs.size(1), device=torch.device("cuda"), dtype=torch.int64)[None, :]

    # Observation dimensions: (batch_size, context_size, ...)
    policy, values = model(obs, positions, block_mask=block_mask)
    log_probs = policy.log_prob(rollout.actions)

    value_pred_clipped = rollout_values + (values - rollout_values).clamp(-vf_clip, vf_clip) 

    value_losses = torch.square(values - rollout.target)
    value_losses_clipped = torch.square(value_pred_clipped - rollout.target)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

    ratio = torch.exp(log_probs - rollout.log_prob)

    advantages = (rollout.advantage - rollout.advantage.mean()) / (rollout.advantage.std() + 1e-8)
    # advantages = rollout.advantage

    loss_actor1 = ratio * advantages
    loss_actor2 = torch.clamp(ratio, 1.0 - vf_clip, 1.0 + vf_clip) * advantages

    actor_loss = -torch.min(loss_actor1, loss_actor2).mean()

    # Entropy regularization
    entropy = policy.entropy()
    entropy_loss = -entropy.mean()

    total_loss = vf_coef * value_loss + actor_loss + entropy_coef * entropy_loss
        # + entropy_coef * entropy_loss

    return total_loss, actor_loss, value_loss

@torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True, disable=False)
def sample_action(model: RLTransformerModel, obs: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    policy, value = model(obs[:, None, ...], positions)
    action: Tensor = policy.sample()
    log_prob = policy.log_prob(action)

    action = action.squeeze(-1)
    log_prob = log_prob.squeeze(-1)
    value = value.squeeze(-1)
    return action, log_prob, value

class Rollout:
    def __init__(self, batch_size: int, trajectory_length: int, obs_dim: int, device: torch.device):
        # observation gets plus one because we need to store the next trailing observation
        self.obs = torch.zeros((batch_size, trajectory_length, obs_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.int64)
        self.reward = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.terminated = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.bool)
        self.truncated = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.bool)

        self.log_prob = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.values = torch.zeros((batch_size, trajectory_length + 1), device=device, dtype=torch.float32)

        self.advantage = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
        self.target = torch.zeros((batch_size, trajectory_length), device=device, dtype=torch.float32)
    
    def calculate_advantage(self):
        values = self.values[:, :-1]
        next_values = self.values[:, 1:]
        self.advantage, self.target = vec_generalized_advantage_estimate(0.97, 0.95, values, next_values, self.reward, self.truncated | self.terminated, self.terminated, time_dim=1)

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
        self.rollout = Rollout(self.batch_size, trajectory_length, obs_dim, device)
        
        # Track cumulative rewards per environment
        self.cumulative_rewards = torch.zeros(self.batch_size, device=device, dtype=torch.float32)
        self.episode_lengths = torch.zeros(self.batch_size, device=device, dtype=torch.int64)
        self.completed_episodes = 0
        self.completed_rewards = []

        self.block_mask = causal_block_mask(trajectory_length)

        self.reward_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        self.terminated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.truncated_tensor = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.obs_tensor = torch.zeros((self.batch_size, obs_dim), device=self.device, dtype=torch.float32)

    def create_rollout(self):
        # self.model.clear_kv_cache()
        # Reset cumulative rewards for new episodes
        self.cumulative_rewards.zero_()
        self.episode_lengths.zero_()
        self.positions.zero_()

        obs, info = self.env.reset()
        obs[..., 1] = 0.0
        obs[..., 3] = 0.0
        self.obs_tensor.copy_(torch.from_numpy(obs))

        for i in range(self.trajectory_length - 1):
            with torch.no_grad():
                action, log_prob, value = sample_action(self.model, self.obs_tensor, self.positions)
            np_action = action.cpu().numpy()
            obs, reward, terminated, truncated, _ = self.env.step(np_action)
            obs[..., 1] = terminated | truncated
            obs[..., 3] = np_action / 2.0
            
            self.reward_tensor.copy_(torch.from_numpy(reward))
            self.terminated_tensor.copy_(torch.from_numpy(terminated))
            self.truncated_tensor.copy_(torch.from_numpy(truncated))

            self.rollout.obs[:, i] = self.obs_tensor
            self.rollout.actions[:, i] = action
            self.rollout.log_prob[:, i] = log_prob
            self.rollout.values[:, i] = value
            self.rollout.reward[:, i] = self.reward_tensor
            self.rollout.terminated[:, i] = self.terminated_tensor
            self.rollout.truncated[:, i] = self.truncated_tensor
            
            self.obs_tensor.copy_(torch.from_numpy(obs))
            self.positions += 1

            # Update cumulative rewards
            self.cumulative_rewards += self.reward_tensor
            self.episode_lengths += 1

            # Track completed episodes
            done_mask = self.terminated_tensor | self.truncated_tensor
            if done_mask.any():
                for env_idx in torch.where(done_mask)[0]:
                    self.completed_episodes += 1
                    self.completed_rewards.append(self.cumulative_rewards[env_idx].item())
                    # Reset rewards and lengths for done environments
                    self.cumulative_rewards[env_idx] = 0
                    self.episode_lengths[env_idx] = 0
        
        with torch.no_grad():
            _, _, value = sample_action(self.model, self.obs_tensor, self.positions)
        self.rollout.values[:, -1] = value

        # Calculate mean reward metrics
        mean_reward = 0.0
        if self.completed_rewards:
            mean_reward = sum(self.completed_rewards) / len(self.completed_rewards)
            # Keep only the most recent 100 episodes
            if len(self.completed_rewards) > 100:
                self.completed_rewards = self.completed_rewards[-100:]

        self.rollout.calculate_advantage()
        return self.rollout, mean_reward



def train():
    torch.set_float32_matmul_precision('high')
    console = Console()

    batch_size = 32
    trajectory_length = 512
    device = torch.device("cuda")

    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    model = RLTransformerModel(
        action_dim=2,
        num_layers=3,
        num_heads=8,
        d_model=256,
        ffn_size=256,
        activation=nn.LeakyReLU(),
        glu=False,
    )
    model.apply(init_weights)
    model.policy_head.linear.weight.data /= 100.0
    # model.value_head.linear.weight.data *= 10.0
    
    model.create_kv_cache(batch_size, trajectory_length, device=device)
    # model.create_block_mask(trajectory_length)

    console.print(f"Model size: {abbreviate_number(get_param_count(model))} parameters")

    total_steps = 4000
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.001, betas=(0.9, 0.9), eps=1e-12)#, weight_decay=0.001, betas=(0.9, 0.9), eps=1e-12)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps, start_factor=1.0, end_factor=0.0)
    model.to(device)

    # model.eval()
    # model(torch.zeros((batch_size, 1, 4), device=device), torch.zeros((batch_size, 1), device=device, dtype=torch.int64))

    trainer = Trainer(model, env, trajectory_length, device)

    # Track cumulative rewards for monitoring
    best_mean_reward = 0.0

    for epoch in track(range(total_steps), console=console, disable=True):
        optimizer.zero_grad()

        if epoch % 4 == 0:
            model.eval()
            rollout, mean_reward = trainer.create_rollout()
        # Debug required grad in the rollout
        # print(f"Rollout: {rollout.obs.requires_grad}, {rollout.actions.requires_grad}, {rollout.reward.requires_grad}, {rollout.terminated.requires_grad}, {rollout.truncated.requires_grad}")
        
        model.train()
        loss, actor_loss, critic_loss = loss_fn(model, rollout, trainer.block_mask)
        # loss.backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        # Update best mean reward
        best_mean_reward = max(best_mean_reward, mean_reward)
        
        # Log metrics
        console.print(f"[Step {epoch}] Loss: {loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Mean Reward: {mean_reward:.2f} | Best Mean Reward: {best_mean_reward:.2f} | Episodes: {trainer.completed_episodes}")
    
    torch.save(model.state_dict(), "pocp3.pth")

def enjoy():
    env = gym.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda")

    model = RLTransformerModel(
        action_dim=2,
        num_layers=6,
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
    obs[..., 1] = 0.0
    obs[..., 3] = 0.0
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    positions = torch.zeros((1, 1), device=device, dtype=torch.int64)

    for _ in range(10000):
        with torch.no_grad():
            action, log_prob, value = sample_action(model, obs_tensor, positions)
        np_action = action.squeeze().cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(np_action)
        obs[..., 1] = terminated | truncated
        obs[..., 3] = np_action / 2.0

        obs_tensor.copy_(torch.from_numpy(obs).unsqueeze(0))
        positions += 1

        if terminated or truncated:
            obs, info = env.reset()
            obs[..., 1] = 0.0
            obs[..., 3] = 0.0
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            positions = torch.zeros((1, 1), device=device, dtype=torch.int64)


if __name__ == "__main__":
    train()
