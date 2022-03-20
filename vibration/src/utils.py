# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:35:06 2022

@author: henry
"""
import numpy as np
import pandas as pd

def split2format(x,y,sampling_rate=3200,feature_name=None):
    # 將一分鐘長度為3200*60的數據,以1秒一次存成一個dataframe,供後續分析
    # 回傳一個儲存dataframe的list,裡面包含有n秒的dataframe
    if feature_name is None:
        feature_name = ["axis1","axis2","axis3"]
    df_data_list = []
    n_sec = len(x) // sampling_rate
    for t in range(n_sec-1):
        x_slice = x[t*sampling_rate:(t+1)*sampling_rate,:]
        df = pd.DataFrame(data=x_slice,columns=feature_name)
        df_data_list.append(df)
    df_label = pd.DataFrame(np.array(y[:len(df_data_list)]).reshape((len(df_data_list),1)),columns=["label"])
    
    return df_data_list, df_label

def save2csv(df_data_list,df_label,prefixes:str):
    for i in range(len(df_data_list)):
        file_name = prefixes + "_" + f"{i}.csv"
        df_data_list[i].to_csv(file_name,index=False)
    
if __name__ == "__main__":
    sec = 65
    axis = 3
    sp = 3200
    x = np.random.random((sp*sec,axis))
    y = np.random.randint(0,2,sec)
    
    df_data_list, df_label = split2format(x,y,sampling_rate=3200,feature_name=None)
    save2csv(df_data_list,df_label,prefixes="move")
                                     