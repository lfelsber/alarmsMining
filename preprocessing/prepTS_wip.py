# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:48:19 2019

@author: lfelsber
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from pandas import DataFrame
from pandas import concat

# import psutil # used to monitore the memory consumption


def series_to_supervised(data, 
                         n_in=1, 
                         n_out=1, 
                         tp=0, 
                         dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
    """
    
    n_in = int(n_in)
    n_out = int(n_out)
    tp = int(tp)
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(tp+0, tp+n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    	else:
    		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
    	agg.dropna(inplace=True)

    return agg


def collapseTimeSlowButMemorySaving(data,var_len):
    c1=time.time()
    vrs=np.arange(1,var_len+1)
    
    exps=[]
    
    for el in vrs:
        exps.append('var'+str(el)+'\(')
        
    df=pd.DataFrame()  
        
    for ex in exps:
        P1Channels = data.filter(regex=ex)
        P1Sum = P1Channels.sum(axis=1)
        
        df=pd.concat([df,P1Sum],axis=1)
    
    c2=time.time()
    print('collapsing took '+str(c2-c1)+' seconds')    
    df.columns=exps
    return df


def collapseTime(data,var_len):
    c1=time.time()
    vrs=np.arange(1,var_len+1)
    
    exps=[]
    
    for el in vrs:
        exps.append('var'+str(el)+'\(')
        
    df=[]
    
    for ex in exps:
        P1Channels = data.filter(regex=ex)
        P1Sum = P1Channels.sum(axis=1)
        
        df.append(P1Sum.tolist())
    
    c2=time.time()
    print('collapsing took '+str(c2-c1)+' seconds')  
    df=np.asarray(df).astype('int32')
    df=pd.DataFrame.from_records(df.T,columns=exps)
    
    return df


def groupTime(data,DTi):

    DTi=int(DTi)
    
    vrs=np.arange(1,DTi+1)
    exps=[]
    for el in vrs:
        exps.append('t-'+str(el)+'\)')
    
    X=[]
    c1=time.time()
    for ex in exps:
        P1Channels = data.filter(regex=ex)
        X.append(P1Channels.values)
        
    X=np.asarray(X).astype('int32')
    c2=time.time()
    print('grouping took '+str(c2-c1)+' seconds')  
    return X.transpose((1,0,2)) 


def convert2DbooleanTo1DInt(matrix):

    int1D=np.zeros(matrix.shape[0])    
    
    for t in range(matrix.shape[0]):
        binary=''
        for j in range(matrix.shape[1]):
            binary+=str(matrix[t,j])
            
        int1D[t]=int(binary,2)
        
    return int1D


def dontFilter(data):
    return data


def filter_priority3(data):
    return data[(data.PRIORITY == 3)]#data[(data.ACTIVE == 'Y')]


def filterFeatures(x_train, x_test, input_signals_list=None, x_test_final=None, threshold=4):
    
    x_train_bin=np.copy(x_train)
    x_train_bin[x_train_bin>0]=1
    xs=np.sum(x_train_bin,axis=(0,1))
    sel=np.where(xs>=threshold)
    
    x_train=x_train[:,:,sel]
    x_train=x_train[:,:,0,:]
    
    x_test=x_test[:,:,sel]
    x_test=x_test[:,:,0,:]
    
    if input_signals_list is not None:
        sel=sel[0].tolist()
        old_list=input_signals_list.copy()
        input_signals_list=[old_list[i] for i in sel]
        input_signals_list_removed=[old_list[i] for i in np.where(xs<threshold)[0].tolist()]
        # input_signals_list=list(np.array(input_signals_list)[sel])
        # input_signals_list_removed=list(np.array(old_list)[sel])
        print('Removing '+str(input_signals_list_removed)+' as they contain less than '+str(threshold)+' alarms/entries')
        
    
    if x_test_final is not None:
        x_test_final=x_test_final[:,:,np.where(xs>threshold)]
        x_test_final=x_test_final[:,:,0,:]
        
    return x_train, x_test, input_signals_list, x_test_final


def testTransform():
    """
    function used to compare two ways to load data
    run the test from terminal using:
    python3.6 -c "from prepTS_wip import *; testTransform()"
    """
    
    def compare(x,y):
        tmp = (x!=y)
        keep = True
        while keep:
            try:
                tmp = sum(tmp)
            except TypeError:
                keep = False
                
        print(tmp)
        return tmp
    
    # parameters explored for the transformation
    seed_range = [1]#[0,1]
    deltaT_range=['12h']
    DTi_range=[32,50,51] # test failed if DTi too big ?
                        # The ratio of class 1, between the old version and new version are different
    DTo_range=[2]#[1,2,3]
    tp_range=[2]#[0,1,2]
    input_variables=['FAULT_MEMBER','FAULT_CODE','ACTIVE']
    output_variables=['FAULT_MEMBER','FAULT_CODE','ACTIVE']
    external_fnames=['../data/external_input/PSB_beam_2h.csv','../data/external_input/PSB_EC_2h.csv']
    add_unix_time=True
    time_frame=['2015-01-01 00:00:00','2018-01-01 00:00:00'] # day_time, use an underscore
    train_test_val=True

    ### parameters for the transformation which are distributed within one job
    p0_target_range = [.9]#[.9,.99]
    testRatio_range = [.5]#[.5,.9]
    y_column_range = [6]#[1,6]

    ### read data
    componentName = "2015start_2017end_PSB_ACTIVE_YES_AND_4_or_more_counts"
    inputDataFile = "../data/real_source_data/"+componentName+".csv"
    data=pd.read_csv(inputDataFile,
                     usecols=['SYSTEM_TS','PRIORITY','ACTIVE','FAULT_FAMILY']+input_variables,
                     compression="zip")

    totalNumberOfTests=len(seed_range)*\
                       len(deltaT_range)*\
                       len(DTi_range)*\
                       len(DTo_range)*\
                       len(tp_range)*\
                       len(p0_target_range)*\
                       len(testRatio_range)*\
                       len(y_column_range)
    
    ###
    # main loop
    ###
    uniqueId = 0 # id of the job to submit
    for seed in seed_range:
        for deltaT in deltaT_range:
            for DTi in DTi_range:
                for DTo in DTo_range:
                    for tp in tp_range:
                        for y_column in y_column_range:
                            for p0_target in p0_target_range:
                                for testRatio in testRatio_range:
                                    uniqueId = uniqueId+1 # id of the job to submit
                                    
                                    print(3*"********\n")
                                    print("starts transformation A")
                                    
                                    t1 = time.time()        
                                    # new way
                                    x_trainA,x_testA,x_tfA,y_trainA,y_testA,y_tfA,alarms_listA = \
                                        transformRealSaveRAMqf(data.copy(),
                                                               deltaT=deltaT,
                                                               DTi=DTi,
                                                               DTo=DTo,
                                                               tp=tp,
                                                               input_variables=input_variables,
                                                               output_variables=output_variables,
                                                               p0_target = p0_target,
                                                               testRatio = testRatio,
                                                               y_column=y_column,
                                                               external_fnames=external_fnames,
                                                               add_unix_time=True,
                                                               add_list_of_signals=True,
                                                               time_frame=time_frame,
                                                               train_test_val=True,
                                                               omit_inactive=True,
                                                               seed = seed)
                                    t2 = time.time()
                                    
                                    print("ends transformation A")
                                    print(3*"********\n")
                                    print("starts transformation B")
                                    
                                    t3 = time.time()
                                    # initial way to load data : need to be replaced by
                                    x_trainB,x_testB,x_tfB,y_trainB,y_testB,y_tfB,alarms_listB = \
                                        transformReal(data.copy(),
                                                      deltaT=deltaT,
                                                      DTi=DTi,
                                                      DTo=DTo,
                                                      tp=tp,
                                                      input_variables=input_variables,
                                                      output_variables=output_variables,
                                                      p0_target = p0_target,
                                                      testRatio = testRatio,
                                                      y_column=y_column,
                                                      external_fnames=external_fnames,
                                                      add_unix_time=True,
                                                      add_list_of_signals=True,
                                                      time_frame=time_frame,
                                                      train_test_val=True,
                                                      omit_inactive=True,
                                                      seed = seed)
                                    t4 = time.time()
                                    
                                    print("ends transformation B")
                                    print(3*"********\n")

                                    print("uniqueId/totalNumberOfTests=",uniqueId,"/",totalNumberOfTests)
                                    print("")
                                    print("seed=",seed)
                                    print("deltaT=",deltaT)
                                    print("DTi=",DTi)
                                    print("DTo=",DTo)
                                    print("tp=",tp)
                                    print("y_column=",y_column)
                                    print("p0_target=",p0_target)
                                    print("testRatio=",testRatio)
                                    
                                    print("A time = ",t2-t1)
                                    print("B time = ",t4-t3)
                                    
                                    tmp=0
                                    tmp+=compare(x_trainA,x_trainB)
                                    tmp+=compare(x_testA,x_testB)
                                    tmp+=compare(x_tfA,x_tfB)
                                    tmp+=compare(y_trainA,y_trainB)
                                    tmp+=compare(y_testA,y_testB)
                                    tmp+=compare(y_tfA,y_tfB)
                                    if tmp!=0:
                                        print("tmp != 0, stopping test")
                                        return tmp
                                    else:
                                        print("passing test !")
                            

def transformRealSaveRAMnoTestRatio(data,
                                    deltaT,
                                    DTi,
                                    DTo,
                                    t_p,
                                    input_variables,
                                    output_variables,
                                    p0_target,
                                    y_column=0,
                                    external_fnames=None,
                                    add_unix_time=False,
                                    time_frame=False,
                                    omit_inactive=True,
                                    seed = 0,
                                    shutdownsFileName=None):
    
    ###
    # reduction of the data to handle
    ###

    # select only active Y
    if omit_inactive==True:
        data=data[(data.ACTIVE == 'Y')]

    data['SYSTEM_TS']=data['SYSTEM_TS'].apply(lambda x: x.rstrip())   
    data['SYSTEM_TS']=pd.to_datetime(data['SYSTEM_TS'])#, format='%d-%b-%y %I.%M.%S.%f %p')
    data.sort_values(by=['SYSTEM_TS'],inplace=True)
    # select only alamr in a given time frame
    if time_frame!=False:
        print(time_frame)
        data=data[(data.SYSTEM_TS > time_frame[0]) & (data.SYSTEM_TS < time_frame[1])]
        
    if shutdownsFileName!=None:
        down_periods=pd.read_csv(shutdownsFileName)
        down_periods['start']=pd.to_datetime(down_periods['start'])
        down_periods['end']=pd.to_datetime(down_periods['end'])
        #iterate over shutdown periods
        for index, row in down_periods.iterrows():
            data=data[(data.SYSTEM_TS < row['start']) | (data.SYSTEM_TS > row['end'])]
    ###
    # reformating some columns
    ###
    
    data['FAULT_FAMILY']=data['FAULT_FAMILY'].apply(lambda x: x.rstrip())
    data['FAULT_MEMBER']=data['FAULT_MEMBER'].apply(lambda x: x.rstrip())
    #data['SOURCE_ID']=data['SOURCE_ID'].apply(lambda x: x.rstrip())

    # getting ready to discretized with respect to time
    data['SYSTEM_TS']=data['SYSTEM_TS'].dt.ceil(deltaT)
    #data.set_index('SYSTEM_TS', drop=False, inplace=True)

    # select only priority 3 data for target variables
    data3=data[(data.PRIORITY == 3)&(data.ACTIVE == 'Y')]
    
    # why do we write a file then read it ?
    counts2d=data.groupby(['SYSTEM_TS']+input_variables).size().astype('int32')
    del data
    counts2d.to_csv('2d_counts_L4.csv',index=True)
    del counts2d    
    df=pd.read_csv('2d_counts_L4.csv', names=['SYSTEM_TS']+input_variables+['COUNT'])
    df['SYSTEM_TS']=pd.to_datetime(df['SYSTEM_TS'])#, format='%d-%b-%y %I.%M.%S.%f %p')        
    df=df.pivot_table(values='COUNT', index='SYSTEM_TS', columns=input_variables,fill_value=0)
    
    # why do we write a file then read it ?
    counts2d3=data3.groupby(['SYSTEM_TS']+output_variables).size().astype('int32')
    del data3
    counts2d3.to_csv('2d_counts3_L4.csv',index=True)
    del counts2d3    
    df3=pd.read_csv('2d_counts3_L4.csv', names=['SYSTEM_TS']+output_variables+['COUNT'])
    df3['SYSTEM_TS']=pd.to_datetime(df3['SYSTEM_TS'])#, format='%d-%b-%y %I.%M.%S.%f %p')
    df3=df3.pivot_table(values='COUNT', index='SYSTEM_TS', columns=output_variables,fill_value=0)

    df=df.resample(deltaT).sum()
    df3=df3.resample(deltaT).sum()
    
    df3tmp = df3.sum() # total number of alarms
    df3tmp = df3tmp.sort_values(ascending=False) # sorting to get the highest number
    df3tmp = df3tmp.reset_index()
    fml=df3tmp.FAULT_MEMBER.tolist()
    if 'member_cons' in fml:
        y_column=df3tmp.index[df3tmp['FAULT_MEMBER']=='member_cons'].tolist()[0]

    tmp=df3tmp.iloc[y_column]
    target_variable = (tmp["FAULT_MEMBER"],tmp["FAULT_CODE"],tmp["ACTIVE"])
    print("element studied \n",tmp)
    print("target variable",target_variable)

    #add unix timestamp
    if add_unix_time==True:
        df['UT']=df.index.astype(np.int64)//10**9#df['SYSTEM_TS']
    
        #add auxiliary time series input data here (e.g. from CALS);
        # input data should be stored in dataframe compatible zip format (timestamps column should be time),
        # then will be resampled to fit main dataframe structure
    
    if external_fnames != None:
        saveRam = True
        if saveRam:
            for fname in external_fnames:
                columns = pd.read_csv(fname,compression='zip', nrows=1).columns.tolist()
                columns.remove('time')
                id = 0
                for column in columns: # loading column by column in order to avoid peak ram consumption
                    id = id+1
                    df_add=pd.read_csv(fname,compression='zip',usecols=['time',column])
                    df_add['time']=pd.to_datetime(df_add['time'])
                    df_add=df_add.set_index('time')
                    df_add=df_add.resample(deltaT).sum()      
                    df=df.join(df_add)
                    df=df.fillna(0)
                    #print('Successfully added '+column+', '+str(id)+' out of '+str(len(columns)))
    
                print('Successfullly added '+fname)
        else:
            for fname in external_fnames:
                df_add=pd.read_csv(fname,compression='zip')
                df_add['time']=pd.to_datetime(df_add['time'])
                df_add=df_add.set_index('time')
                df_add=df_add.resample(deltaT).sum()      
                df=df.join(df_add)
                df=df.fillna(0)
                print('Successfullly added '+fname)
        try:    
            del df_add
        except:
            print('no ext file')


    # print(psutil.virtual_memory()) # used to monitore the memory consumption
    signals_list=list(df)
    

    data_superv3=series_to_supervised(df3,
                                      0,
                                      DTo,
                                      t_p)
    
    # data_superv3=data_superv3[data_superv3.index>=data_superv.index[0]] 
    #set start time to earliest possible time where past input observed
    # data_superv=data_superv[data_superv.index<=data_superv3.index[-1]] 
    #set end time to latest possible time where future output was observed
    # data_superv=data_superv[data_superv.index>=data_superv3.index[0]] 
    #set start time to earliest possible time where prio 3 problems were pobserved

    #collapse time for target data
    Y=collapseTime(data_superv3,df3.shape[1])
    del data_superv3
    Y.columns=list(df3)
    # del df3
    
    Y = Y[target_variable] # choose only one column
    Y = np.asarray(Y)
    print("checking if the number of fault is correct wrt to the target variable chosen:",sum(Y))
    Y[Y>1]=1 # force to only one alarm in the given interval


    
    #Subsampling for more balanced class distribution
    np.random.seed(seed)
    p1_target=1-p0_target
    p1=np.sum(Y)/len(Y) #current fraction of class 1
    len_class_1=p1*len(Y) #number of class1 members
    len_class_0_target=np.round(p0_target*len_class_1/p1_target).astype(int) #number of class 0 with target fraction

    idx_1s=np.arange(len(Y))[Y==1] #location of class 1
    idx_0s=np.arange(len(Y))[Y==0] #location of class 0
    idx_1sm1=idx_1s-1 #location one before class 1
    idx_1sm1=idx_1sm1[idx_1sm1>=0] #making sure to stay within array limits
    idx_1sm2=idx_1s-2 #location two before class 1
    idx_1sm2=idx_1sm2[idx_1sm2>=0] #making sure to stay within array limits
    idx_1sp1=idx_1s+1 #location one after class 1
    idx_1sp1=idx_1sp1[idx_1sp1<len(Y)] #making sure to stay within array limits
    idx_1sp2=idx_1s+2 #location two after class 1
    idx_1sp2=idx_1sp2[idx_1sp2<len(Y)] #making sure to stay within array limits
    idx_0s_target=np.random.choice(idx_0s,size=len_class_0_target) #subsample class 0
    
    idx_subsampled=np.concatenate((idx_0s_target,idx_1s,idx_1sm1,idx_1sm2,idx_1sp1,idx_1sp2)) #concatenate locations
    idx_subsampled=np.unique(idx_subsampled) #optional: dont take any datum twice
    idx_subsampled=np.sort(idx_subsampled)

    subsampled_times = df3.index[idx_subsampled] #get timestamps of subsampled items
    
    #generate times of subsampled inpt windows
    X_times=[]
    for i in subsampled_times:
        X_time=[]
        start_time = i - pd.to_timedelta(deltaT) - DTi * pd.to_timedelta(deltaT) # tacking into account tp ?
        end_time = i - pd.to_timedelta(deltaT)
        X_time.append(start_time)
        X_time.append(end_time)
        X_times.append(X_time)
        
    #check which input windows actually contain data/are within considered timeframe (and drop if not)
    i=0
    drop_inidces=[]
    X_times_dropped=X_times[:]
    for el in X_times:
        if el[0]<df.index[0]:
            #drop item
            X_times_dropped.remove(el)
            drop_inidces.append(i)
            print('Removed time '+str(el[0]))
        i+=1
        
    mask=np.ones(len(idx_subsampled), dtype=bool)
    mask[drop_inidces] = False
    idx_subsampled = idx_subsampled[mask]
    X_times=X_times_dropped[:]

    # print(psutil.virtual_memory()) # used to monitore the memory consumption
    
    #compile inputs in X array
    X=np.ndarray((len(idx_subsampled),DTi,df.shape[1]))
    df['SYSTEM_TS']=df.index
    i=0
    for X_time in X_times:
        X[i] = df[(df.SYSTEM_TS > X_time[0]) & (df.SYSTEM_TS <= X_time[1])].to_numpy()[:,:-1]
        i+=1
    
    X_s = np.flip(X,axis=1)
    Y_s=Y[idx_subsampled] #select subsampled output data

    print("X_s="+str(X_s.shape))
    print("Y_s="+str(Y_s.shape))

    return X_s,Y_s,X_times,signals_list,target_variable
    

def divideWRTtestRatio(X_s,
                       Y_s,
                       testRatio,
                       train_test_val):
        
    if train_test_val==True:                   
        x_train1, x_test_final, y_train1, y_test_final = train_test_split(X_s, Y_s, test_size=0.2, shuffle=False)
        x_train, x_test, y_train, y_test = train_test_split(x_train1, y_train1, test_size=testRatio, shuffle=False)
        return x_train,x_test, x_test_final,y_train,y_test, y_test_final
        
    else:
        x_train, x_test, y_train, y_test = train_test_split(X_s, Y_s, test_size=0.2, shuffle=False)
        return x_train,x_test,None,y_train,y_test,None

    
def transformRealSaveRAMnoTestRatioPEMS(data, # WIP
                                    deltaT,
                                    DTi,
                                    DTo,
                                    t_p,
                                    p0_target,
                                    y_column=0,
                                    external_fnames=None,
                                    add_unix_time=False,
                                    time_frame=False,
                                    omit_inactive=True,
                                    seed = 0,
                                    trigger_traffic=0.3,
                                    filter_by_corr=True,
                                    remove_target_from_input=True,
                                    add_weekday=True):
    
    ###
    # reduction of the data to handle
    ###

    data['time']=pd.to_datetime(data['time'])#, format='%d-%b-%y %I.%M.%S.%f %p')
    data.sort_values(by=['time'],inplace=True)
    
    # select only alamr in a given time frame
    if time_frame!=False:
        print(time_frame)
        data=data[(data.time > time_frame[0]) & (data.time < time_frame[1])]
        
    data['time']=data['time'].dt.ceil(deltaT)
    data.set_index('time', drop=True, inplace=True)

    data3=data[data > trigger_traffic]
    data3=data3.fillna(0)
    data3[data3>0]=1
    #select sensors where traffic jams are less frequent (needs manual adjustment)
    data3=data3.loc[:, (data3.sum(axis=0) >= 100)]
    data3=data3.loc[:, (data3.sum(axis=0) <= 5000)]
    data3=data3.loc[:, (data3.var(axis=0) > 0)]
    
    data=data.resample(deltaT).sum()
    data3=data3.resample(deltaT).sum()
    target_variable=list(data3)[y_column]
    
    print('filtered outputs')
    
    
    #check which columns are covariated and select those for input
    if filter_by_corr==True:
        corr_mat=data.corr()[target_variable]
        corr_v=corr_mat.nlargest(n=50)
        v_list=corr_v.index.tolist() #get indices of 50 most correlated variables
        if remove_target_from_input:
            v_list.remove(target_variable) #make task harder and insight more valuable
        data=data[v_list]
        
    if add_weekday==True:
        data['weekday']=data.index
        data['weekday']=data['weekday'].dt.dayofweek
        #one-hot-encode
        data = pd.concat([data,pd.get_dummies(data['weekday'], prefix='day')],axis=1)
        data.drop(['weekday'],axis=1, inplace=True)
        
    print('filterd inputs')

    #add unix timestamp
    if add_unix_time==True:
        data['UT']=data.index.astype(np.int64)//10**9#df['SYSTEM_TS']
    
        #add auxiliary time series input data here (e.g. from CALS);
        # input data should be stored in dataframe compatible zip format (timestamps column should be time),
        # then will be resampled to fit main dataframe structure


    # print(psutil.virtual_memory()) # used to monitore the memory consumption
    signals_list=list(data)
    

    data_superv3=series_to_supervised(data3,
                                      0,
                                      DTo,
                                      t_p)
    
    # data_superv3=data_superv3[data_superv3.index>=data_superv.index[0]] 
    #set start time to earliest possible time where past input observed
    # data_superv=data_superv[data_superv.index<=data_superv3.index[-1]] 
    #set end time to latest possible time where future output was observed
    # data_superv=data_superv[data_superv.index>=data_superv3.index[0]] 
    #set start time to earliest possible time where prio 3 problems were pobserved

    #collapse time for target data
    Y=collapseTime(data_superv3,data3.shape[1])
    del data_superv3
    Y.columns=list(data3)
    # del df3
    
    Y = Y[target_variable] # choose only one column
    Y = np.asarray(Y)
    print("checking if the number of fault is correct wrt to the target variable chosen:",sum(Y))
    Y[Y>1]=1 # force to only one alarm in the given interval


    
    #Subsampling for more balanced class distribution
    np.random.seed(seed)
    p1_target=1-p0_target
    p1=np.sum(Y)/len(Y) #current fraction of class 1
    len_class_1=p1*len(Y) #number of class1 members
    len_class_0_target=np.round(p0_target*len_class_1/p1_target).astype(int) #number of class 0 with target fraction

    idx_1s=np.arange(len(Y))[Y==1] #location of class 1
    idx_0s=np.arange(len(Y))[Y==0] #location of class 0
    idx_1sm1=idx_1s-1 #location one before class 1
    idx_1sm1=idx_1sm1[idx_1sm1>=0] #making sure to stay within array limits
    idx_1sm2=idx_1s-2 #location two before class 1
    idx_1sm2=idx_1sm2[idx_1sm2>=0] #making sure to stay within array limits
    idx_1sp1=idx_1s+1 #location one after class 1
    idx_1sp1=idx_1sp1[idx_1sp1<len(Y)] #making sure to stay within array limits
    idx_1sp2=idx_1s+2 #location two after class 1
    idx_1sp2=idx_1sp2[idx_1sp2<len(Y)] #making sure to stay within array limits
    idx_0s_target=np.random.choice(idx_0s,size=len_class_0_target) #subsample class 0
    
    idx_subsampled=np.concatenate((idx_0s_target,idx_1s,idx_1sm1,idx_1sm2,idx_1sp1,idx_1sp2)) #concatenate locations
    idx_subsampled=np.unique(idx_subsampled) #optional: dont take any datum twice
    idx_subsampled=np.sort(idx_subsampled)

    subsampled_times = data3.index[idx_subsampled] #get timestamps of subsampled items
    
    #generate times of subsampled inpt windows
    X_times=[]
    for i in subsampled_times:
        X_time=[]
        start_time = i - pd.to_timedelta(deltaT) - DTi * pd.to_timedelta(deltaT) # tacking into account tp ?
        end_time = i - pd.to_timedelta(deltaT)
        X_time.append(start_time)
        X_time.append(end_time)
        X_times.append(X_time)
        
    #check which input windows actually contain data/are within considered timeframe (and drop if not)
    i=0
    drop_inidces=[]
    X_times_dropped=X_times[:]
    for el in X_times:
        if el[0]<data.index[0]:
            #drop item
            X_times_dropped.remove(el)
            drop_inidces.append(i)
            print('Removed time '+str(el[0]))
        i+=1
        
    mask=np.ones(len(idx_subsampled), dtype=bool)
    mask[drop_inidces] = False
    idx_subsampled = idx_subsampled[mask]
    X_times=X_times_dropped[:]

    # print(psutil.virtual_memory()) # used to monitore the memory consumption
    
    #compile inputs in X array
    X=np.ndarray((len(idx_subsampled),DTi,data.shape[1]))
    data['SYSTEM_TS']=data.index
    i=0
    for X_time in X_times:
        X[i] = data[(data.SYSTEM_TS > X_time[0]) & (data.SYSTEM_TS <= X_time[1])].to_numpy()[:,:-1]
        i+=1
    
    X_s = np.flip(X,axis=1)
    Y_s=Y[idx_subsampled] #select subsampled output data

    print("X_s="+str(X_s.shape))
    print("Y_s="+str(Y_s.shape))

    return X_s,Y_s,X_times,signals_list,target_variable
    
