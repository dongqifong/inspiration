# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:29:43 2022

@author: henry
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ModelDataset(Dataset):
    def __init__(self, df_list):
        self.df_list = df_list
        self.label = np.array([1] * len(self.df_list))
        
    def __getitem__(self,idx):
        x = self.df_list[idx].values # nd.array:(256, 3)
        x = torch.tensor(x).float()
        x = torch.transpose(x, 0, 1) # nd.array:(3, 256)
        label = self.label[idx]
        return x, label
    
    def __len__(self):
        return len(self.df_list)
    
def get_dataloader(df_list,label=None,batch_size=1,shuffle=False):
    dataset = ModelDataset(df_list)
    if label is not None:
        dataset.label = label
    model_loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return model_loader



if __name__ == "__main__":
    import pandas as pd
    df_list = []
    for i in range(10):
        x = np.random.random((256,3))
        df = pd.DataFrame(x,columns=["x","y","z"])
        df_list.append(df)
    print(len(df_list))
    print(df_list[0])
    dataset = ModelDataset(df_list)
    x,label = dataset.__getitem__(0)
    print(x.shape)
    print(label)
    data_loader = get_dataloader(df_list,batch_size=5,shuffle=False)
