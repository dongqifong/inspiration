import torch
import numpy as np
import make_dataset


def predict_clf(model, x, y=None, index=None, batch_size=1):
    model.eval()
    dataloader = make_dataset.get_loader(x, y, index, batch_size)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (x, y, index) in enumerate(dataloader):
            out = model(x)
            y_pred.append(out.numpy().argmax(axis=1))
            y_true.append(y)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_pred, y_true
