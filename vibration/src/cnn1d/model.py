# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:31:27 2022

@author: henry
"""

import torch
import torch.nn as nn


class ModelCNN1d(nn.Module):
    def __init__(self):
        # assume input_size = 3200
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=128,stride=32) # (16,97)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32,stride=4) # (32,17)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16,stride=2) # (64,1)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=3)
        
    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        
        x = self.conv3(x)
        x = nn.ReLU()(x)
        
        
        code = x.view(-1,x.shape[1]*x.shape[2])
        
        x = nn.Dropout(p=0.2)(code)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        
        x = nn.Dropout(p=0.2)(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        
        x = nn.Dropout(p=0.2)(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        
        x = self.fc4(x)
        return code, x
    
if __name__ == "__main__":
    x = torch.randn((2,1,3200))
    model = ModelCNN1d()
    code, y_pred = model(x)
    print(code.shape)
    print(y_pred.shape)