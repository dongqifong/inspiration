# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:35:16 2022

@author: henry
"""

import torch.nn as nn

class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self,y_pred,y_true):
        
        return nn.CrossEntropyLoss()(y_pred,y_true)