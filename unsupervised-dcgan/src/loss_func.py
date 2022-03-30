# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:05:26 2022

@author: henry
"""

import torch
import torch.nn as nn


def fraud_loss(d_fake):
    # to train generator
    # to fool discriminator
    # d_fake: D(G(z))
    device = d_fake.device
    n = d_fake.shape[0]
    alpha = torch.ones((n,1)).float().to(device)
    return nn.BCELoss()(d_fake,alpha)

def apparent_loss(x,x_):
    d = x - x_
    return torch.norm(d,p=1)

def latent_loss(z,z_):
    d = z - z_
    return torch.norm(d,p=1)

class GLoss(nn.Module):
    def __init__(self,wf=0.33,wa=0.33,wl=0.33):
        super().__init__()
        self.wf = wf
        self.wa = wa
        self.wl = wl
    def forward(self,d_fake,x,x_,z,z_):
        lf = fraud_loss(d_fake)
        la = apparent_loss(x, x_)
        ll = latent_loss(z, z_)
        return self.wf*lf + self.wa*la + self.wl*ll
        
class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,d_real, d_fake):
        device = d_real.device
        n = d_real.shape[0]
        real_label = torch.ones((n,1)).float().to(device)
        fake_label = torch.zeros((n,1)).float().to(device)
        d_loss = nn.BCELoss()(d_real,real_label) + nn.BCELoss()(d_fake,fake_label)
        return d_loss / 2
    

def anomaly_score(x,x_,z,z_):
    return apparent_loss(x,x_) + latent_loss(z,z_)
    

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    
    d_real = nn.Sigmoid()(torch.randn((10,1)).to(device))
    d_fake = nn.Sigmoid()(torch.randn((10,1)).to(device))
    dloss = DLoss()
    loss_d = dloss(d_real, d_fake)
    print(loss_d)
    
    gloss = GLoss()
    x = torch.randn((10,2,100)).to(device)
    x_ = torch.randn((10,2,100)).to(device)
    z = torch.randn((10,3,20)).to(device)
    z_= torch.randn((10,3,20)).to(device)
    loss_g = gloss(d_fake,x,x_,z,z_)
    print(loss_g)
    
    print(anomaly_score(x,x_,z,z_))
    print(apparent_loss(x,x_))
    
    
