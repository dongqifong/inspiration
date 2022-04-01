# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:43:43 2022

@author: henry
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src import loss_func, makedataset, model


def train_generator(model_G, model_D, train_loader, optimizer_G, loss_G,loss_D):
    # x_, z, z_ = model_G(x)
    g_loss = 0.0
    d_loss = 0.0
    for batch_idx, (x,_) in enumerate(train_loader):
        optimizer_G.zero_grad()
        x_, z, z_ = model_G(x)
        
        d_real = model_D(x)
        d_fake = model_D(x_)
        
        dloss = loss_D(d_real, d_fake)
        gloss = loss_G(d_fake,x,x_,z,z_)
        
        gloss.backward()
        optimizer_G.step()
        
        d_loss += dloss.item()
        g_loss += gloss.item()
    return round(g_loss / (batch_idx+1),5), round(d_loss / (batch_idx+1),5)

def train_discriminator(model_G, model_D, train_loader, optimizer_D, loss_G, loss_D):
    # x_, z, z_ = model_G(x)
    g_loss = 0.0
    d_loss = 0.0
    for batch_idx, (x,_) in enumerate(train_loader):
        optimizer_D.zero_grad()
        x_, z, z_ = model_G(x)
        
        d_real = model_D(x)
        d_fake = model_D(x_)
        
        dloss = loss_D(d_real, d_fake)
        gloss = loss_G(d_fake,x,x_,z,z_)
        
        dloss.backward()
        optimizer_D.step()
        
        d_loss += dloss.item()
        g_loss += gloss.item()
    return round(g_loss / (batch_idx+1),5), round(d_loss / (batch_idx+1),5)


class Handler():
    def __init__(self,model_G,model_D,train_loader,turn_period=5):
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
        self.d_count = turn_period
        
        
    def train(self, epochs):
        
        for epoch in range(epochs):
            if self.g_count>0:
                # print("train G",end="\r")
                g_loss, d_loss = train_generator(self.model_G, self.model_D, self.train_loader, self.optimizer_G, self.loss_G, self.loss_D)
                self.g_loss_hist.append(g_loss)
                self.d_loss_hist.append(d_loss)
                
                
            if self.d_count>0:
                # print("train D",end="\r")
                g_loss, d_loss = train_discriminator(self.model_G, self.model_D, self.train_loader, self.optimizer_D, self.loss_G, self.loss_D)
                self.g_loss_hist.append(g_loss)
                self.d_loss_hist.append(d_loss)
                
                
            self.g_count -= 1
            self.d_count -= 1
                
            if self.g_count==0 or self.d_count==0:
                self.g_count = self.g_count * -1
                self.d_count = self.d_count * -1
                
            print(f"[{epoch}]/[{epochs}],generator loss:{self.g_loss_hist[-1]}, discriminator loss:{self.d_loss_hist[-1]}", end="\r")
            
        return None
    
    def predict(self, df_list, batch_size=5):
        data_loader = makedataset.get_dataloader(df_list, batch_size=batch_size, shuffle=False)
        arr_x = []
        arr_x_reconst = []
        arr_z = []
        arr_z_ =[ ]
        arr_anomaly_s = []
        with torch.no_grad():
            for batch_idx, (x,_) in enumerate(data_loader):
                x_, z, z_ = self.model_G(x)
                anomaly_s = loss_func.anomaly_score(x,x_,z,z_)
                
                arr_x.append(x.cpu().numpy())
                arr_x_reconst.append(x_.cpu().numpy())
                arr_z.append(z.cpu().numpy())
                arr_z_.append(z_.cpu().numpy())
                arr_anomaly_s.append(anomaly_s.cpu().numpy())
                
        arr_x  = np.concatenate(arr_x,axis=0)
        arr_x_reconst  = np.concatenate(arr_x_reconst,axis=0)
        arr_z  = np.concatenate(arr_z,axis=0)
        arr_z_  = np.concatenate(arr_z_,axis=0)
        arr_anomaly_s = np.concatenate(arr_anomaly_s,axis=0)
        
        result = {}
        result["x"] = arr_x
        result["x_reconst"] = arr_x_reconst
        result["z"] = arr_z
        result["z_"] = arr_z_
        result["anomaly_score"] = arr_anomaly_s
        return  result
            
if __name__ =="__main__":
    generator = model.Generator()
    discriminator = model.Discriminator()
    
    x = torch.randn((128,3,256))
    x_, z, z_ = generator(x)
    print(x_.shape)
    print(z.shape)
    print(z_.shape)
    
    d_real = discriminator(x)
    d_fake = discriminator(x_)
    print(d_real.shape)
    print(d_fake.shape)
    
    df_list = []
    for i in range(10):
        x = np.random.random((256,3))
        df = pd.DataFrame(x,columns=["x","y","z"])
        df_list.append(df)

    train_loader = makedataset.get_dataloader(df_list,batch_size=5)
    
    model_handler = Handler(model_G=generator, model_D=discriminator, train_loader=train_loader)
    model_handler.train(10)
    
    print(model_handler.g_loss_hist)
    print(model_handler.d_loss_hist)
    
    result = model_handler.predict(df_list,batch_size=10)
    print(result["x"].shape)
    print(result["x_reconst"].shape)
    print(result["z"].shape)
    print(result["z_"].shape)
    print(result["anomaly_score"])

