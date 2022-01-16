import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ModelDataset(Dataset):
    def __init__(self, x: np.ndarray, y=None) -> None:
        super().__init__()
        self.x = x
        if y is None:
            self.y = np.array([-1]*len(x))
        else:
            self.y = y

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx]).float()
        y = torch.tensor(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)


def get_loader(x: np.ndarray, y: np.ndarray, batch_szie=128, shuffle=False):
    dataset = ModelDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_szie, shuffle=shuffle)
    return dataloader
