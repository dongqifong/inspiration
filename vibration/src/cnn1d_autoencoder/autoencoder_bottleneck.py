# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 23:08:48 2022

@author: henry
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=141,groups=3,kernel_size=64,stride=8)
        self.conv2 = nn.Conv1d(in_channels=141, out_channels=64, kernel_size=32,stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16,stride=2)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1,stride=1)
        
        

        self.convt1 = nn.ConvTranspose1d(in_channels=3, out_channels=32, kernel_size=16,stride=2,output_padding=1)
        self.convt2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=32,stride=4,output_padding=1)
        self.convt3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=64,stride=8)
        self.convt4 = nn.ConvTranspose1d(in_channels=128, out_channels=3, kernel_size=1,stride=1)
        
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        
        
        x = self.conv2(x)
        x = self.leakyrelu(x)
        
        
        x = self.conv3(x)
        x = self.leakyrelu(x)
        
        
        x = self.conv4(x)
        x = self.leakyrelu(x)
        code = x
         
        
        
        x = self.convt1(x)
        x = self.leakyrelu(x)
        
        x = self.convt2(x)
        x = self.leakyrelu(x)
        
        
        x = self.convt3(x)
        x = self.leakyrelu(x)
        
        
        x = self.convt4(x)
        
        return code, x
        
        
def dot_sum(code):
    device = code.device
    n = code.shape[0]
    s = Variable(torch.tensor(0.0)).to(device)
    # rec = []
    for i in range(n):
        # rec.append(code[i])
        v1 = code[i][0]
        v2 = code[i][1]
        v3 = code[i][2]
        dot_sum = (torch.dot(v1, v2).pow(2) + torch.dot(v2, v3).pow(2) + torch.dot(v1, v3).pow(2))/3
        s = s + dot_sum
    return s
        
        
device = "cuda:0"
model = Model(input_channels=3).to(device)
x = torch.randn((10,3,1600)).to(device)
code, y = model(x)
print(code.shape)

s = dot_sum(code)
print(s)