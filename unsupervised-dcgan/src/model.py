# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:03:34 2022

@author: henry
"""

import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self,input_channels, out_channels, kernel_size, stride, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=padding, groups=groups,bias=False)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x
    
        
class Encoder(nn.Module):
    def __init__(self,input_channels:int):
        super().__init__()
        # assume: 
            # input_channels:3
            # input_size:512
        self.conv1 = CNNBlock(input_channels=input_channels, out_channels=240, kernel_size=32, stride=16, groups=3) # (-1,240,61)
        self.conv2 = nn.Conv1d(in_channels=240,out_channels=12,kernel_size=8,stride=2,bias=False)
        
    def forward(self,x):
        x = self.conv1(x)
        z = self.conv2(x)
        return z
    
class Decoder(nn.Module):
    def __init__(self,input_channels:int):
        super().__init__()
        # assume: 
            # input_channels:3
            # input_size:512
        self.conv1 = nn.Sequential(nn.ConvTranspose1d(in_channels=12, out_channels=240, kernel_size=8,stride=2,output_padding=1,bias=False),
                                   nn.BatchNorm1d(240),
                                   nn.ReLU(),
                                   nn.ConvTranspose1d(in_channels=240, out_channels=3, kernel_size=32,stride=16,groups=3,bias=False),
                                   nn.BatchNorm1d(3))
    def forward(self,x):
        x_ = self.conv1(x)
        return x_   

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder(3)
        self.decoder = Decoder(240)
        self.encoder2 = Encoder(3)
        
    def forward(self,x):
        z = self.encoder1(x)
        x_ = self.decoder(z)
        z_ = self.encoder2(x_)
        return x_, z, z_
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Encoder(3)
        
        self.clf = nn.Sequential(nn.Linear(in_features=48, out_features=24,bias=False),
                                 nn.ReLU(),
                                 nn.Linear(in_features=24, out_features=2,bias=False),
                                 nn.Sigmoid()
                                 )
        
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,48)
        y = self.clf(x)
        return y
      
if __name__ == "__main__":
    x = torch.randn((10,3,256))
    encoder = Encoder(3)
    z = encoder(x)
    print(z.shape)
    
    decoder = Decoder(240)
    x_ = decoder(z)
    print(x_.shape)
    
    disc = Discriminator()
    y = disc(x)
    print(y.shape)
    
    generator = Generator()
    x_, z, z_ = generator(x)
    print(x_.shape)
    print(z.shape)
    print(z_.shape)    
    
    



