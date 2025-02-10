import torch
from torch.utils.data import Dataset
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        numpy_dataset = np.load(file_path)
        self.tokens = torch.from_numpy(numpy_dataset["tokens"]).to(torch.int16)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        return tokens
