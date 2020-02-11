#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:45:17 2019

@author: lfelsber
"""
from joblib import Parallel, delayed
import pytimber
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

def dosome(ib1,now_minus_a_day,now,deltaT):
    
    ldb = pytimber.LoggingDB() #initialize pytimber interface
    print('Loading '+ib1)
    data=ldb.get(ib1,now_minus_a_day,now) #load data
#    data=ldb.getScaled(ib1,now_minus_a_day,now,scaleAlgorithm='REPEAT', scaleSize='1', scaleInterval='MINUTE') #too slow
    tt,vv=data[ib1] #extract from data structure
    if len(vv)>0: #check if containing data
        tt=list(tt) #convert to list for easier handling
        tt.insert(0,now_minus_a_day) #insert timestamp at beginning of interval (padding)
        tt.append(now) #insert timestampt at end of interval (padding)
        vv=list(vv) #convert to list for easier handling
        vv.insert(0,vv[0]) #insert value at beginning of interval (padding)
        vv.append(vv[-1])   #insert value at end of interval (padding)
        
    else:
        tt=[now_minus_a_day,now] #set dummy timestamps
        vv=[-1234,-1234] #set dummy values
        
    df = pd.DataFrame({'time':tt, ib1:np.nan_to_num(vv)}) #convert into pandas dataframe for easier subsampling
    df['time']=pd.to_datetime(df['time'], unit='s') #convert date string to datetime format
    df=df.set_index('time') 
#    df_len=len(df)
    
#    if ib1[-5:]=='STATE':   #specific to power converter signal - convert string states to numeric values
#        mapping = {'7':-5,'ABORTING':-4.,'FLT_OFF':-3.,'FLT_STOPPING':-2.5,'OFF':-2.,'STARTING':-1.,'TO_STANDBY':-1.5,'ON_STANDBY':0.,'IDLE': 1., 'ARMED': 2.,'STOPPING':2.5, 'RUNNING': 3.}
#        df=df.replace({ib1:mapping})
        
    df = df.loc[~df.index.duplicated(keep='last')]
    df.index.names = ['Date']
    df=df.resample(deltaT).sum().astype('int16') #resample amount of alarms in time window
##    df=df.fillna(0.)#fill NaNs with zeros
#    
#    df_h_mean=dfr.resample('5000T').mean() #get mean
##    df_h_mean=df_h_mean.fillna(method='pad')
#    df_h_max=dfr.resample('5000T').max() #get max
##    df_h_max=df_h_max.fillna(method='pad')
#    df_h_min=dfr.resample('5000T').min() #get min
##    df_h_min=df_h_min.fillna(method='pad')
#    
#    del(dfr)
#    
#    print('Compression factor is '+str(df_len/(1e-9+len(df_h_max)+len(df_h_mean)+len(df_h_min))*100)+'%.\n')
    
#    plt.figure(figsize=(4,3))
#    plt.plot(df,'.',label='data')
#    plt.plot(df_h_mean,'-x',label='mean')
#    plt.plot(df_h_max,'-o',label='max')
#    plt.plot(df_h_min,'-*',label='min')
#    plt.legend()
#    plt.xlabel('Tim
#e')
#    plt.title(ib1)
#    
    return df#,df_h_mean,df_h_max,df_h_min


def getCALSdat(variable_list,t1,t2,deltaT='1h',parallel_loaders=1,to_csv=False):

    res=[]
    res.append(Parallel(n_jobs=parallel_loaders, prefer="threads")(delayed(dosome)(el,t1,t2,deltaT) for el in variable_list)) #fetch data in parallel
    df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), res[0])

#    for el in variable_list:
#        res.append(dosome(el,t1,t2,deltaT)) #fetch data in parallel
#    df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), res)
#    
    
    del res
    df_final.index.names=['time']
    print('Took '+str(time.time()-mtn)+' seconds.')
    
    #for el in res[0]:
    #
    #    plt.figure(figsize=(3,2))
    #    plt.plot(el[0],'.',label='data',alpha=0.3)
    #    plt.plot(el[1],'-',label='mean',alpha=0.3)
    #    plt.plot(el[2],'--',label='max',alpha=0.3)
    #    plt.plot(el[3],'-.',label='min',alpha=0.3)
    #    plt.legend()
    #    plt.xlabel('Time')
    #    plt.title(list(el[3])[-1])
    
    if to_csv==True:
        df_final.to_csv('CALS.csv',index=True,compression='zip')
    
    return df_final


def getCALSdatMemSav(variable_list,t1,t2,deltaT='1h',to_csv=False):

    df=pd.DataFrame()
#    res.append(Parallel(n_jobs=parallel_loaders, prefer="threads")(delayed(dosome)(el,t1,t2,deltaT) for el in variable_list)) #fetch data in parallel
#    df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), res[0])

    for el in variable_list:
        dfn=dosome(el,t1,t2,deltaT)#fetch data
        df=pd.concat([df, dfn], axis=1)
    
    df.index.names=['time']
    print('Took '+str(time.time()-mtn)+' seconds.')
    
    #for el in res[0]:
    #
    #    plt.figure(figsize=(3,2))
    #    plt.plot(el[0],'.',label='data',alpha=0.3)
    #    plt.plot(el[1],'-',label='mean',alpha=0.3)
    #    plt.plot(el[2],'--',label='max',alpha=0.3)
    #    plt.plot(el[3],'-.',label='min',alpha=0.3)
    #    plt.legend()
    #    plt.xlabel('Time')
    #    plt.title(list(el[3])[-1])
    
    if to_csv==True:
        df.to_csv('CALS.csv',index=True,compression='zip')
    
    return df

'''main'''


mtn=time.time() #get current time to see how long data fetching takes

ldb = pytimber.LoggingDB() #initialize pytimber interface (check documentation)

now= mtn #set end time of data to be fetched
now_minus_a_day = now - 3600*24*365*5   #set start time of data to be fetched
#ib1="LHC.BCTDC.A6R4.B1:BEAM_INTENSITY"
#ibs='RPMBB.RR57.RSS.A56B1:I_DIFF_MA'

#ibs=ldb.search('RPMBB.RR57.RSS.%:I_E%') #define variables to load in some way
ibs=ldb.search('PSB.EC:%' )
#ibs=ibs+ibs2  #define variables to load in some way
#with open('vars_small.csv', newline='') as csvfile:
#    vars_smallcsv = list(csv.reader(csvfile))
#ibs=vars_smallcsv[100]

#dat=getCALSdat(ibs,now_minus_a_day,now,'10S',4,True)
dat=getCALSdatMemSav(ibs,now_minus_a_day,now,'4S',True)

#import seaborn as sns
##visualize
#df_mat=np.array(dat)
#df_mat_log=np.log(df_mat+1)
#plt.figure()
#sns.heatmap(df_mat, annot=False, vmax=3., cmap="YlGnBu")
#plt.yticks(np.arange(0.5, len(dat.index), 1)[::1500], dat.index[::1500])
#plt.tight_layout()
##plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
#del df_mat, df_mat_log





