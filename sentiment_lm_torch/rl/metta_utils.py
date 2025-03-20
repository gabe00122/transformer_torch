import torch
import torch.nn as nn

OBS_NORMALIZATIONS = {
    'agent': 1,
    'agent:group': 10,
    'agent:hp': 30,
    'agent:frozen': 1,
    'agent:energy': 255,
    'agent:orientation': 1,
    'agent:shield': 1,
    'agent:color': 255,
    'agent:inv:ore': 100,
    'agent:inv:battery': 100,
    'agent:inv:heart': 100,
    'agent:inv:laser': 100,
    'agent:inv:armor': 100,
    'agent:inv:blueprint': 100,
    'inv:ore': 100,
    'inv:battery': 100,
    'inv:heart': 100,
    'inv:laser': 100,
    'inv:armor': 100,
    'inv:blueprint': 100,
    'wall': 1,
    'wall:hp': 30,
    'generator': 1,
    'generator:hp': 30,
    'generator:ready': 1,
    'mine': 1,
    'mine:hp': 30,
    'mine:ready': 1,
    'altar': 1,
    'altar:hp': 30,
    'altar:ready': 1,
    'armory': 1,
    'armory:hp': 30,
    'armory:ready': 1,
    'lasery': 1,
    'lasery:hp': 30,
    'lasery:ready': 1,
    'lab': 1,
    'lab:hp': 30,
    'lab:ready': 1,
    'factory': 1,
    'factory:hp': 30,
    'factory:ready': 1,
    'temple': 1,
    'temple:hp': 30,
    'temple:ready': 1,
    'last_action': 10,
    'last_action_argument': 10,
    'agent:kinship': 10,
    'hp': 30,
    'ready': 1,
    'converting': 1,
}

class ObservationNormalizer(nn.Module):
    def __init__(self, grid_features: list[str]):
        super().__init__()

        num_objects = len(grid_features)

        obs_norm = torch.tensor([OBS_NORMALIZATIONS[k] for k in grid_features], dtype=torch.float32)
        obs_norm = obs_norm.view(1, num_objects, 1, 1)

        self.register_buffer('obs_norm', obs_norm, persistent=False)

    def forward(self, obs):
        return obs / self.obs_norm
