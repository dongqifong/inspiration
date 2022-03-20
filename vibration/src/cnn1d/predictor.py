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
    code_list = []
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(data_loader):
            code,y_pred = model(x)
            code_list.append(code.numpy())
            y_pred_list.append(y_pred.numpy().argmax(axis=1))
            y_true_list.append(y.numpy())
    code_arr = np.concatenate(code_list,axis=0)
    y_pred_arr = np.concatenate(y_pred_list,axis=0)
    y_true_arr = np.concatenate(y_true_list,axis=0)
    return code_arr, y_pred_arr, y_true_arr