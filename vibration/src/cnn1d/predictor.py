# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:32:04 2022

@author: henry
"""
import numpy as np
import torch

def predict(model,data_loader):
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(data_loader):
            y_pred = model(x)
            y_pred_list.append(y_pred.numpy())
            y_true_list.append(y.numpy())
    return np.concatenate(y_pred_list,axis=0), np.concatenate(y_true_list,axis=0)