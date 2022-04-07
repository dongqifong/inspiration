# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ModelDataset(Dataset):
    def __init__(self,df_x, y=None):
        # df: pd.Dataframe
        # y: np.ndarray, shape = (n,1)
        super().__init__()
        self.x = df_x.values
        if y is None:
            self.y = np.zeros(len(self.x))
        else:
            self.y = y
        self.max = np.max(self.x,axis=0)
        self.min = np.min(self.x,axis=0)
        
    def __getitem__(self, index):
        x = (self.x[index] - self.min) / (self.max - self.min)
        x = torch.tensor(x).float()
        
        y = self.y[index]
        y = torch.tensor(y).long()
        return x, y
    
    def __len__(self):
        return len(self.x)
    
def get_loader(df_x,y=None,batch_size=128,shuffle=False):
    dataset = ModelDataset(df_x,y)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader
        
        

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=input_size),
                                 nn.LeakyReLU())
        
        self.fc2 = nn.Sequential(nn.Linear(in_features=input_size, out_features=6),
                                 nn.LeakyReLU())
        
        self.fc3 = nn.Sequential(nn.Linear(in_features=6, out_features=2))
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.dropout(x)
        code = self.fc2(x)
        x = self.fc3(code)
        return code, x
    
def loss_func(y_pred, y_true, code):
    cross_en = nn.CrossEntropyLoss()(y_pred, y_true)
    class0 = torch.where(y_true==0)
    class1 = torch.where(y_true==1)
    std0 = torch.sum(torch.std(code[class0],axis=0))
    std1 = torch.sum(torch.std(code[class1],axis=0))
    center0 = torch.mean(code[class0],axis=0)
    center1 = torch.mean(code[class1],axis=0)
    distance = torch.norm(center0-center1)
    return cross_en + 0.1*std0 + 0.1*std1 + 0.1*(1/distance+1e-4)

def train_one_epoch(model,data_loader,optimizer):
    running_loss = 0.0
    for batch_idx, (x,y_true) in enumerate(data_loader):
        optimizer.zero_grad()
        code, y_pred = model(x)
        loss = loss_func(y_pred, y_true, code)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (batch_idx+1)


class Trainer:
    def __init__(self,model,train_loader):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.train_loss_hist = []
        self.show_period = 5
        
    def train(self,epochs):
        for epoch in range(epochs):
            running_loss = train_one_epoch(self.model,self.train_loader,self.optimizer)
            self.train_loss_hist.append(running_loss)
            
            if epoch==0 or (epoch+1)%self.show_period==0:
                print(f"{epoch+1}/{epochs+1}, running_loss={self.train_loss_hist[-1]}",end="\r")
        return None
    
    def save_model(self,model_name,save_dir):
        save_path = Path(save_dir) / (model_name + ".pth")
        torch.save(self.model, str(save_path))
        print(f"Model is place at {str(save_path)}")
        return None
        

        
def predictor(df_x,model,y=None):
    data_loader = get_loader(df_x,y=y,batch_size=128,shuffle=False)
    model.eval()
    code_arr = []
    y_true_arr = []
    y_pred_arr = []
    with torch.no_grad():
        for batch_idx, (x,y_true) in enumerate(data_loader):
            code, y_pred = model(x)
            y_true_arr.append(y_true.numpy())
            code_arr.append(code.numpy())
            y_pred_arr.append(y_pred.numpy().argmax(axis=1))
    code_arr = np.concatenate(code_arr,axis=0)
    y_true_arr = np.concatenate(y_true_arr,axis=0)
    y_pred_arr = np.concatenate(y_pred_arr,axis=0)
    return code_arr, y_true_arr, y_pred_arr
            
            

df_x = pd.DataFrame(data=np.random.random((256,9)))
y = np.random.choice([0,1],256)

model = Model(input_size=9)
train_loader = get_loader(df_x,y)
trainer = Trainer(model, train_loader)
trainer.train(1)

code_arr, y_true_arr, y_pred_arr=predictor(df_x,model,y=None)

