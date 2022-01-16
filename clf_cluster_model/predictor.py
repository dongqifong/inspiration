import numpy as np
import torch
import torch.nn as nn
from make_dataset import get_loader
from model_pool import ClfCluster


def predict(model: nn.Module, x: np.ndarray, y=None, batch_size=128):
    dataloader = get_loader(x, y, batch_size)
    y_pred = []
    y_true = []
    code = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            code_, out = model(x)

            y = y.numpy()
            out = out.numpy()
            code_ = code_.numpy()

            y_true.append(y)
            y_pred.append(np.argmax(out, axis=1))
            code.append(code_)
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0), np.concatenate(code, axis=0)


if __name__ == "__main__":
    x = np.random.random((2, 60, 1500))
    y = np.array([0, 1])
    m = ClfCluster(60, 1500, 5)

    y_true, y_pred, code = predict(model=m, x=x, batch_size=128)
    print(y_true)
    print(y_pred)
    print(code.shape)
