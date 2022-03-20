# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:31:55 2022

@author: henry
"""
from pathlib import Path

import torch
import torch.optim as optim

from src.cnn1d import loss_func


def train_one_epoch(model,data_loader,loss_func,optimizer,device="cpu"):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (x,y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        code, y_pred = model(x)
        loss = loss_func(y_pred,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return round(running_loss / (batch_idx+1),5)
        
def valid_one_epoch(model,data_loader,loss_func,device="cpu"):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            code, y_pred = model(x)
            loss = loss_func(y_pred,y)
            running_loss += loss.item()
    return round(running_loss / (batch_idx+1),5)
        
def show_training_loss(epoch,epochs,train_loss_hist,valid_loss_hist=None):
    if valid_loss_hist is not None:
        print(f"[{epoch+1}/{epochs}], training_loss:{train_loss_hist[-1]}, valid_loss:{valid_loss_hist[-1]}")
    else:
        print(f"[{epoch+1}/{epochs}], training_loss:{train_loss_hist[-1]}")
    

class Trainer:
    def __init__(self,model,train_loader,valid_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        
        self.train_loader = train_loader
        self.train_loss_hist = []
        
        if valid_loader is not None:
            self.valid_loader = valid_loader
            self.valid_loss_hist = []
        else:
            self.valid_loader = None
            self.valid_loss_hist = None
            
        self.optimizer = optim.Adam(model.parameters(),lr=1e-4)
        self.loss_func = loss_func.LossFunc()
        
        self.show_period = 5
        
    def train(self,epochs):
        for epoch in range(epochs):
            if self.valid_loader is not None:
                running_loss = valid_one_epoch(self.model, self.valid_loader, self.loss_func,self.device)
                self.valid_loss_hist.append(running_loss)
            running_loss = train_one_epoch(self.model, self.train_loader, self.loss_func, self.optimizer,self.device)
            self.train_loss_hist.append(running_loss)
            if (epoch+1)%self.show_period==0 or epoch==0:
                show_training_loss(epoch,epochs, self.train_loss_hist, self.valid_loss_hist)
        return None
            
            
    def save_model(self,save_dir,model_name:str):
        model_path_state_dict = Path(save_dir) / (model_name + "_state_dict.pth")
        model_path_full = Path(save_dir) / (model_name + "_full.pth")
        self.model.cpu()
        self.model.eval()
        torch.save(self.model.state_dict(),model_path_state_dict)
        torch.save(self.model,model_path_full)
        return None
            
            
            
            
            
            
            
            
            
            
            
                