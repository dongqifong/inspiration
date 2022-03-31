# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:43:43 2022

@author: henry
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src import loss_func, makedataset


def train_generator(model_G, model_D, train_loader):
    g_loss = 0
    return g_loss

def train_discriminator(model_G, model_D, train_loader):
    d_loss = 0
    return d_loss

def generator_no_grad(model_G, model_D, train_loader):
    g_loss = 0
    return g_loss

def discriminator_no_grad(model_G, model_D, train_loader):
    d_loss = 0
    return d_loss

class Handler():
    def __init__(self,model_G,model_D,train_loader):
        self.model_G = model_G
        self.model_D = model_D
        self.train_loader = train_loader
        self.optimizer_G = optim.Adam(model_G.parameters(),lr=1e-4)
        self.optimizer_D = optim.Adam(model_D.parameters(),lr=1e-4)
        
        self.loss_G = loss_func.GLoss()
        self.loss_D = loss_func.DLoss()
        
        self.g_loss_hist = []
        self.d_loss_hist = []
        
        self.show_period = 5
        
        self.g_count = 0
        self.d_count = 5
        
        self.exchange = 1
        
    def train(self, epochs):
        
        for epoch in range(epochs):
            if self.g_count>0:
                g_loss = train_generator(self.model_G, self.model_D, self.train_loader)
                self.g_loss_hist.append(g_loss)
                
                d_loss = discriminator_no_grad(self.model_G, self.model_D, self.train_loader)
                self.d_loss_hist.append(d_loss)
                
            if self.d_count>0:
                d_loss = train_discriminator(self.model_G, self.model_D, self.train_loader)
                self.d_loss_hist.append(d_loss)
                
                g_loss = generator_no_grad(self.model_G, self.model_D, self.train_loader)
                self.g_loss_hist.append(g_loss)
                
            self.g_count -= 1
            self.d_count -= 1
                
            if self.g_count==0 or self.d_count==0:
                self.g_count = self.g_count * -1
                self.d_count = self.d_count * -1
            
        return None
    
    def predict(self, df_list):
        dataloader = makedataset.get_dataloader(df_list, batch_size=5, shuffle=False)
        pass
            
    
    