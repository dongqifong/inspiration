# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 00:14:25 2022

@author: henry
"""

import numpy as np


from src.cnn1d import model,trainer,make_dataset

n_data = 30
train_x = np.random.random((n_data,3200,3))
train_y = np.random.randint(0,3,n_data)

n_data = 10
valid_x = np.random.random((n_data,3200,3))
valid_y = np.random.randint(0,3,n_data)

train_loader = make_dataset.get_loader(train_x,train_y,batch_size=15,shuffle=False)
valid_loader = make_dataset.get_loader(valid_x,valid_y,batch_size=5,shuffle=False)

model_cnn1d = model.ModelCNN1d()
trainer_cnn1d = trainer.Trainer(model=model_cnn1d, train_loader=train_loader,valid_loader=valid_loader)


trainer_cnn1d.train(10)

