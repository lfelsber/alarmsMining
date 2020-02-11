# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:03:19 2020

@author: lfelsber
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale(x,scaler):

    dt=x.shape[0]
    dti=x.shape[1]
    ds=x.shape[2]
    
    x=x.reshape((dt*dti,ds))    
    x = scaler.transform(x)
    x_scaled=x.reshape((dt,dti,ds))

    return x_scaled

def trainScaler(x,min=0,max=1):
    
    dt=x.shape[0]
    dti=x.shape[1]
    ds=x.shape[2]
    
    x=x.reshape((dt*dti,ds))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(min, max))
    scaler = scaler.fit(x)
    #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print the first 5 rows
    x = scaler.transform(x)
    x_scaled=x.reshape((dt,dti,ds))
    
    return x_scaled, scaler

def unscale(x_scaled,scaler):
    
    dt=x_scaled.shape[0]
    dti=x_scaled.shape[1]
    ds=x_scaled.shape[2]
    
    x_scaled=x_scaled.reshape((dt*dti,ds))
    x_scaled = scaler.inverse_transform(x_scaled)
    x=x_scaled.reshape((dt,dti,ds))
    
    return x
    