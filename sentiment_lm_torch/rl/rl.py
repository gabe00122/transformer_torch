import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution
from torch.nn.attention.flex_attention import BlockMask
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
import gymnasium as gym
from mettagrid.gym_wrapper import make, MultiToDiscreteWrapper

from einops import rearrange

from rich.console import Console
from rich.progress import track

from sentiment_lm_torch.model.transformer import TransformerLayer
from sentiment_lm_torch.model.util import init_weights
from sentiment_lm_torch.rl.metta_utils import ObservationNormalizer
from sentiment_lm_torch.utils import get_param_count, abbreviate_number
from sentiment_lm_torch.model.attention import causal_block_mask

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
def loss_fn(model: RLTransformerModel, rollout: 'Rollout', block_mask: BlockMask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vf_coef = 0.480
    entropy_coef = 0.00209

    vf_clip = 0.1

    obs = rollout.obs
    # rollout_values = rollout.values[:, :-1]

    positions = torch.arange(obs.size(1), device=torch.device("cuda"), dtype=torch.int64)[None, :]

    # Observation dimensions: (batch_size, context_size, ...)
    policy, values = model(obs, positions, block_mask=block_mask)
    log_probs = policy.log_prob(rollout.actions)

    # value_pred_clipped = rollout_values + (values - rollout_values).clamp(-vf_clip, vf_clip) 

    value_losses = torch.square(values - rollout.target)
    # value_losses_clipped = torch.square(value_pred_clipped - rollout.target)
    # value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    value_loss = 0.5 * value_losses.mean()

    ratio = torch.exp(log_probs - rollout.log_prob)

    loss_actor1 = ratio * rollout.advantage
    loss_actor2 = torch.clamp(ratio, 1.0 - vf_clip, 1.0 + vf_clip) * rollout.advantage

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
    
    def calculate_advantage(self):
        values = self.values[..., :-1]
        next_values = self.values[..., 1:]
        self.advantage, self.target = vec_generalized_advantage_estimate(0.986, 0.818, values, next_values, self.reward, self.truncated | self.terminated, self.terminated, time_dim=1)
        
        # Normalize advantage
        self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-8)


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

        self.rollout.calculate_advantage()
        return self.rollout, self.rollout.reward.sum()


def train():
    torch.set_float32_matmul_precision('high')
    console = Console()

    env = MultiToDiscreteWrapper(make("bases", overrides=["game.num_agents=28", "game.max_steps=256"]))
    image_shape = env.unwrapped.single_observation_space.shape
    action_dim = env.action_space.n
    num_agents = env.unwrapped.num_agents

    # image_shape = (image_shape[1], image_shape[2], image_shape[0])
    print(image_shape)
    print(action_dim)
    print(num_agents)

    batch_size = num_agents
    trajectory_length = 512
    device = torch.device("cuda")

    d_model = 512
    observation_encoder = MetaCnnEncoder(
        img_shape=image_shape,
        d_model=d_model,
        activation=nn.LeakyReLU(),
        channels=[32, 64],
        grid_features=env.unwrapped.grid_features
    )
    policy_head = PolicyHead(d_model=d_model, action_dim=2)
    value_head = ValueHead(d_model=d_model)

    model = RLTransformerModel(
        num_layers=2,
        num_heads=8,
        d_model=d_model,
        ffn_size=d_model,
        activation=nn.LeakyReLU(),
        glu=False,
        observation_encoder=observation_encoder,
        policy_head=policy_head,
        value_head=value_head,
    )
    model.apply(init_weights)
    nn.init.orthogonal_(model.policy_head.linear.weight)
    model.policy_head.linear.weight.data *= 0.01
    nn.init.orthogonal_(model.value_head.out_linear.weight)

    # +1 because we need to store the next trailing observation for the value head
    model.create_kv_cache(batch_size, trajectory_length + 1, device=device)

    # console.print(f"Cnn Size: {abbreviate_number(get_param_count(model.layers))}")
    console.print(f"Model size: {abbreviate_number(get_param_count(model))} parameters")
    
    total_steps = 100_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00141)#, weight_decay=0.001, betas=(0.9, 0.9), eps=1e-12)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps, start_factor=1.0, end_factor=0.0)
    model.to(device)

    trainer = Trainer(model, env, trajectory_length, device)

    # Track cumulative rewards for monitoring
    best_mean_reward = 0.0

    for epoch in track(range(total_steps), console=console, disable=True):
        optimizer.zero_grad()

        if epoch % 3 == 0:
            model.eval()
            rollout, mean_reward = trainer.create_rollout()
        
        # breakpoint()

        model.train()
        loss, actor_loss, critic_loss = loss_fn(model, rollout, trainer.block_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        
        # Update best mean reward
        best_mean_reward = max(best_mean_reward, mean_reward)
        
        # Log metrics
        console.print(f"[Step {epoch}] Loss: {loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Mean Reward: {mean_reward:.4f} | Best Mean Reward: {best_mean_reward:.4f} | Episodes: {trainer.completed_episodes}")
    
    torch.save(model.state_dict(), "metta_simple.pth")

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


if __name__ == "__main__":
    train()
