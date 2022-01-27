import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_loader, optimizer, loss_func, valid_loader: None) -> None:
        self.show_period = 1
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loss = []
        self.valid_loss = []
        pass

    def train(self, epochs=1):
        for epoch in range(epochs):
            if self.valid_loader is not None:
                self.valid_loss.append(valid_one_epoch(
                    self.model, self.valid_loader, self.loss_func))

            self.train_loss.append(train_one_epoch(
                self.model, self.valid_loader, self.loss_func, self.optimizer))

            if epoch % self.show_period == 0 or epoch == 0:
                if self.valid_loader:
                    show_progress(
                        epoch, epochs, self.train_loss[-1], self.valid_loss[-1])
                else:
                    show_progress(
                        epoch, epochs, self.train_loss[-1])
        return None

    def save_model(self, model_name):
        save_model(self.model, model_name)


def show_progress(epoch=None, epochs=None, train_loss=None, valid_loss=None):
    if valid_loss is not None:
        print(
            f"[{epoch}/{epochs}], training_loss:[{train_loss}], valid_loss:[{valid_loss}]", end="\r")
    else:
        print(
            f"[{epoch}/{epochs}], training_loss:[{train_loss}]", end="\r")


def save_model(model, model_name: str, mode="full"):
    if mode == "all":
        torch.save(model, model_name+"_full_model.pth")
    else:
        torch.save(model.state_dict(), model_name+"_state_dict.pth")


def train_one_epoch(model, train_loader, loss_func, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        out = model(x)
        optimizer.zero_grad()
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    return round(running_loss / (batch_idx+1), 5)


def valid_one_epoch(model, valid_loader, loss_func):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_loader):
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()
            running_loss = running_loss + loss.item()
    return round(running_loss / (batch_idx+1), 5)
