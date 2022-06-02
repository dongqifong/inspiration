# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 23:20:54 2022

@author: henry
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 8
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=9, stride=4, groups=self.in_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels//2, kernel_size=7, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels//2))
        
        self.out_channels = self.out_channels//2
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels//2, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels//2))
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels,out_channels=self.in_channels*2,kernel_size=5,stride=2,output_padding=1),
            nn.BatchNorm1d(self.in_channels*2),
            nn.ReLU())
        
        self.in_channels = self.in_channels*2
        
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels,out_channels=self.in_channels*2,kernel_size=7,stride=3,output_padding=2),
            nn.BatchNorm1d(self.in_channels*2),
            nn.ReLU())
        
        self.in_channels = self.in_channels*2
        
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.in_channels,out_channels=self.out_channels*2,kernel_size=9,stride=4,output_padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=1))
        
    def forward(self,x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x
    
class Generator(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.encoder1 = Encoder(in_channels)
        self.encoder2 = Encoder(in_channels)
        self.decoder1 = Decoder(in_channels=2, out_channels=in_channels)
        
    def forward(self,x):
        z1 = self.encoder1(x)
        x_reconst = self.decoder1(z1)
        z2 = self.encoder2(x_reconst)
        return x_reconst, z1, z2
    
class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.clf = nn.Sequential(nn.Linear(in_features=36, out_features=18),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_features=18, out_features=9),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_features=9, out_features=2),
                                 nn.Sigmoid())
        
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,36)
        x = self.clf(x)
        return x
        

if __name__ == "__main__":
    n_channels = 1
        
    x = torch.randn((20,n_channels,512))
        
    e = Encoder(n_channels)
    out_e = e(x)
    print(out_e.shape)
        
    d = Decoder(in_channels=2, out_channels=n_channels)
    out_d = d(out_e)
    print(out_d.shape)
    
    gen = Generator(n_channels)
    x_reconst, z1, z2 = gen(x)
    print(x_reconst.shape)
    print(z1.shape)
    print(z2.shape)
    
    dis = Discriminator(n_channels)
    out_dis = dis(x)
    print(out_dis.shape)
