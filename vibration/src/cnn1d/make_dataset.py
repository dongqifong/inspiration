# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:31:33 2022

@author: henry
"""
import torch
from torch.utils.data import Dataset, DataLoader

class CNN1dDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self,idx):
        x = torch.tensor(self.x[idx]).float()
        y = torch.tensor(self.y[idx]).long()
        
        return x, y
    
    def __len__(self):
        return len(self.x)
    
def get_loader(x,y,batch_size=1,shuffle=False):
    dataset = CNN1dDataset(x,y)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader