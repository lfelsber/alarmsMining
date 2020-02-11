# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:43:28 2020

@author: lfelsber
"""

import pandas as pd

beam_dest=pd.read_csv('../data/external_input/PSB_beam_1h.csv',compression='zip')
beam_dest['time']=pd.to_datetime(beam_dest['time'])
beam_dest.set_index('time', inplace=True)
inactive_duration_sampling='5d' #assume 5 days is long enough for a technical stop but longer than severe failure

beam_dest['sum']=beam_dest.sum(axis=1)
beam_dest=beam_dest[['sum']]


beam_dest_res=beam_dest.resample(inactive_duration_sampling).sum()
beam_dest_res['start_time']=beam_dest_res.index
beam_dest_res['end_time']=beam_dest_res.index.shift(1)
beam_dest_res=beam_dest_res[beam_dest_res['sum']==0]
beam_dest_res.drop('sum',axis=1,inplace=True)
# beam_dest_res['start_time']=pd.to_datetime(beam_dest_res['start_time'])
# beam_dest_res['end_time']=pd.to_datetime(beam_dest_res['end_time'])
beam_dest_res['start_interval']=beam_dest_res['start_time']-beam_dest_res['end_time'].shift(1)
beam_dest_res['end_interval']=beam_dest_res['start_interval'].shift(-1)
beam_dest_res['start_interval'][beam_dest_res['start_interval']!=pd.to_timedelta('0')]=True
beam_dest_res['end_interval'][beam_dest_res['end_interval']!=pd.to_timedelta('0')]=True

off_s=beam_dest_res.values

off_times=[]
off_time=[]
for i in range(off_s.shape[0]):

    #add start time minus one day (to avoid shutting machine down affects)
    if off_s[i,2]==True:
        off_time.append(off_s[i,0]-pd.Timedelta('1d'))
        
    if off_s[i,3]==True:
        off_time.append(off_s[i,1])
        off_times.append(off_time)
        off_time=[]
        
        
off_times=pd.DataFrame(off_times,columns=['start','end'])
off_times.to_csv('../data/external_input/shutdowns.csv',index=False)