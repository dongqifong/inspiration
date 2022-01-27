import torch
from torch.utils.data import Dataset, DataLoader


class ModelDataset(Dataset):
    def __init__(self, x, y=None, index=None) -> None:
        super().__init__()

        self.x = torch.tensor(x).float()

        if y is None:
            self.y = torch.zeros((len(self.x),), dtype=torch.int)
        else:
            self.y = torch.tensor(y)

        if index is None:
            self.index = [i for i in range(len(x))]
        else:
            self.index = index

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y, self.index[index]

    def __len__(self):
        return len(self.x)


def get_loader(x, y=None, index=None, batch_size=1, shuffle=False):
    dataset = ModelDataset(x, y, index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
