import torch


def predict(model, x, y=None):
    model.eval()
