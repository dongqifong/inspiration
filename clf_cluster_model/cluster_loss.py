import torch
import torch.nn as nn


class ClusterLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, label, n_class):
        loss_all = 0.0
        for i in range(n_class):
            mask = torch.where(label == i)
            xi = x[mask]
            print(xi)
            xi_std = torch.std(xi)
            print(xi_std)
            loss_all = loss_all + xi_std
        return loss_all


if __name__ == "__main__":
    x = torch.tensor([[0, 1, 2], [0, 2, 4], [1, 2, 3], [4, 5, 6]]).float()
    label = torch.tensor([0, 1, 0, 1])
    n_class = 2
    Loss = ClusterLoss()
    loss = Loss(x, label, n_class)
    print(loss)
