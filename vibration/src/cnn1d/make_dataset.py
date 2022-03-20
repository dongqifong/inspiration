# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:31:33 2022

@author: henry
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CNN1dDataset(Dataset):
    def __init__(self,x_arr,label=None):
        super().__init__()
        # x_arr: (n, 3200, 3)
        self.x_arr = x_arr # (n,3200,3)
        self.x_arr = np.transpose(self.x_arr,(0,2,1)) # (n,3,3200)
        if label is not None:
            self.label = label
        else:
            self.label = np.array([-1]*len(self.df_x_list))
        
    def __getitem__(self,idx):
        x = torch.tensor(self.x_arr[idx]).float()
        y = torch.tensor(self.label[idx]).long()
        return x, y
    
    def __len__(self):
        return len(self.x_arr)
    
def get_loader(x,label=None,batch_size=1,shuffle=False):
    dataset = CNN1dDataset(x,label)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader