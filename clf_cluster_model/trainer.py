import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cluster_loss import ClusterLoss


def train_one_epoch(model: nn.Module, optimizer: optim, train_loader: DataLoader, **kwargs):
    n_class = kwargs["n_class"]
    alpha = 0.05

    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        code, out = model(x)
        cross_entropy_loss = nn.CrossEntropyLoss()(out, y)
        cluster_loss = ClusterLoss()(code, y, n_class)
        loss_sum = cross_entropy_loss + alpha*cluster_loss
        loss_sum.backward()
        optimizer.step()
        running_loss = running_loss + loss_sum.item()
    return running_loss/(batch_idx+1)


def valid_one_epoch(model: nn.Module, valid_loader: DataLoader, **kwargs):
    n_class = kwargs["n_class"]
    alpha = 0.05
    model.valid()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_loader):
            code, out = model(x)
            cross_entropy_loss = nn.CrossEntropyLoss()(out, y)
            cluster_loss = ClusterLoss()(code, y, n_class)
            loss_sum = cross_entropy_loss + alpha*cluster_loss
            running_loss = running_loss + loss_sum.item()
        return running_loss/(batch_idx+1)


def show_progress(epoch, epochs, train_loss, valid_loss):
    print(
        f"{epoch+1}/{epochs}, train_loss:{train_loss[-1]}, valid_loss:{valid_loss[-1]}", end="\r")


class Trainer:
    def __init__(self, model: nn.Module, train_loader, valid_loader) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_loss = []
        self.valid_loss = []
        self.show_period = 10
        pass

    def train(self, epochs):
        for epoch in range(epochs):
            valid_loss = valid_one_epoch(
                self.model, self.valid_loader, n_class=5)
            self.valid_loss.append(valid_loss)

            train_loss = train_one_epoch(self.model, self.optimizer,
                                         self.train_loader, n_class=5)
            self.train_loss.append(train_loss)
            if (epoch+1) % self.show_period == 0 or epoch == 0:
                show_progress(epoch, epochs, self.train_loss, self.valid_loss)
