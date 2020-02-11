# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:39:37 2020

@author: lfelsber
"""

import pandas as pd
import numpy as np

import sys

sys.path.insert(1, '../artificialData/')
import artificialData as ad

sys.path.insert(1, '../preprocessing/')
import prepTS_wip as pts
import scalers as sc

sys.path.insert(1, '../input_activation/')
import inputActivation as iA

rootDirLib='../TSC_benchmark/dl-4-tsc-master/'
sys.path.insert(1,rootDirLib)
import main as main

import svm as svm


### parameters for the generation of fake data
hour=3600
day=24*hour
year=365.35*day
# durationOfSeries_range = [2.7*year]
durationOfSeries = 2.7*year
distributionLaw = 'normal'
seed=0

### 2 sys data
numberOfPrecursors_range = [2]
delayParameterBetweenPrecursorsMEAN_range = [day]
delayParameterBetweenPrecursorsSTD_range = [hour]
meanDurationOfPrecursorsMEAN = 5
meanDurationOfPrecursorsSTD = 0
delayParameterBetweenLastPrecursorAndEventMEAN_range = [10*hour]
delayParameterBetweenLastPrecursorAndEventSTD_range = [hour]
meanDurationOfConsequenceMEAN = 6
meanDurationOfConsequenceSTD = 0
delayParameterBetweenEventAndFirstPrecursorMEAN_range = [0.1*year]
delayParameterBetweenEventAndFirstPrecursorSTD_range = [0.1*year]
precursorName="prec"
consequenceName="cons"

### 1 sys data
delayParameterBetweenEventsMEAN_range = [0.04*year]
delayParameterBetweenEventsSTD_range = [0.04*year]
meanDurationOfEventsMEAN = 6
meanDurationOfEventsSTD = 0
number_of_noise_channels_range=[1,2,4,8,16,32,64,128,256,512]

alarmLevel = 2

### parameters for the transformation                        
deltaT_range=['2h','3h']
DTi_range=[40]
DTo_range=[1,2,3,4]
tp_range=[0]
input_variables=[ 'FAULT_MEMBER', # not relevant in this script
                    # 'FAULT_CODE', # not relevant in this script
                    'ACTIVE'] # not relevant in this script
output_variables=['FAULT_MEMBER', # not relevant in this script
                    'FAULT_CODE', # not relevant in this script
                    'ACTIVE'] # not relevant in this script
p0_target_range = [0.8]
testRatio_range = [.50,.45,.40,.35,.30,.25,.20, 0.]
y_column_range = [0]
rootDirData = '../data/syntheticExperiment'

external_fnames=None
time_frame=['2015-01-01 00:00:00','2018-01-01 00:00:00']
train_test_val=True
data_mode='artificial'

### parameters for the training
classifier_name_range = ['svm','random_forest','knn','fcn','fcn_dropout_twice','fcn_dropout_extreme','cnn']
nb_epochs=2000


uniqueId = 0
# for durationOfSeries in durationOfSeries_range:
for numberOfPrecursors in numberOfPrecursors_range:
    for delayParameterBetweenPrecursorsMEAN in delayParameterBetweenPrecursorsMEAN_range:
        for delayParameterBetweenPrecursorsSTD in delayParameterBetweenPrecursorsSTD_range:
            for delayParameterBetweenLastPrecursorAndEventMEAN in delayParameterBetweenLastPrecursorAndEventMEAN_range:
                for delayParameterBetweenLastPrecursorAndEventSTD in delayParameterBetweenLastPrecursorAndEventSTD_range:
                    for delayParameterBetweenEventAndFirstPrecursorMEAN in delayParameterBetweenEventAndFirstPrecursorMEAN_range:
                        for delayParameterBetweenEventAndFirstPrecursorSTD in delayParameterBetweenEventAndFirstPrecursorSTD_range:

                            ###
                            ### generation of fake data
                            ###
                            initialData = ad.artificialDataCreation2systemsDatetime(
                                durationOfSeries = durationOfSeries,
                                numberOfPrecursors = numberOfPrecursors,
                                delayParameterBetweenPrecursors = [delayParameterBetweenPrecursorsMEAN,
                                                                   delayParameterBetweenPrecursorsSTD],
                                meanDurationOfPrecursors = [meanDurationOfPrecursorsMEAN,
                                                            meanDurationOfPrecursorsSTD],
                                delayParameterBetweenLastPrecursorAndEvent = [delayParameterBetweenLastPrecursorAndEventMEAN,
                                                                            delayParameterBetweenLastPrecursorAndEventSTD],
                                meanDurationOfConsequence = [meanDurationOfConsequenceMEAN,
                                                            meanDurationOfConsequenceSTD],
                                delayParameterBetweenEventAndFirstPrecursor = [delayParameterBetweenEventAndFirstPrecursorMEAN,
                                                                               delayParameterBetweenEventAndFirstPrecursorSTD],
                                distributionLaw = distributionLaw,
                                fileName = None, # data not stored as the generation is almost instantaneous
                                seed=seed,
                                precursorName=precursorName,
                                consequenceName=consequenceName)

                            for delayParameterBetweenEventsMEAN in delayParameterBetweenEventsMEAN_range :
                                for delayParameterBetweenEventsSTD in delayParameterBetweenEventsSTD_range:
                                    for number_of_noise_channels in number_of_noise_channels_range:
                                        eventName_range = ["prec"+str(i) for i in range(number_of_noise_channels)]
                                        for eventName in eventName_range:

                                            initialData1sys = ad.artificialDataCreation1systemDatetime(
                                                durationOfSeries = durationOfSeries,
                                                delayParameterBetweenEvents = [delayParameterBetweenEventsMEAN,
                                                                                delayParameterBetweenEventsSTD],
                                                meanDurationOfEvents = [meanDurationOfEventsMEAN,
                                                                       meanDurationOfEventsSTD],
                                                distributionLaw = distributionLaw,
                                                fileName = None,
                                                seed = None,
                                                eventName = eventName,
                                                alarmLevel = alarmLevel)   
                                            initialData = pd.concat([initialData,initialData1sys])

                                        #print(initialData2sys.shape)
                                        #print(initialData1sys.shape)
                                        
                                        initialData.sort_values("SYSTEM_TS", axis=0, ascending=True, inplace=True) 
                                        print(initialData.shape)

                                        for deltaT in deltaT_range:
                                            for DTi in DTi_range:
                                                for DTo in DTo_range:
                                                    for tp in tp_range:
                                                        for y_column in y_column_range:
                                                            for p0_target in p0_target_range:
                                                            
                                                                if data_mode == 'artificial':
                                                                    data = initialData.copy()
                                                                elif data_mode == 'artificial_and_real':
                                                                    data=pd.read_csv('../data/real_source_data/2014_2017_contains_3L4.csv')
                                                                    data=pd.concat([data,initialData])
                                                                elif data_mode == 'real':
                                                                    data=pd.read_csv('../data/real_source_data/2014_2017_contains_3L4.csv')
                                                                
                                                                ### transform data
                                                                X_s,Y_s,X_times,signals_list,target_variable = \
                                                                            pts.transformRealSaveRAMnoTestRatio(data.copy(),
                                                                                                                deltaT=deltaT,
                                                                                                                DTi=DTi,
                                                                                                                DTo=DTo,
                                                                                                                t_p=tp,
                                                                                                                input_variables=input_variables,
                                                                                                                output_variables=output_variables,
                                                                                                                p0_target = p0_target,
                                                                                                                y_column=y_column,
                                                                                                                external_fnames=external_fnames,
                                                                                                                add_unix_time=False,
                                                                                                                time_frame=time_frame)
                                                                
                                                                # signals_list=[signals_list,target_variable,X_times]
                                                                for testRatio in testRatio_range:        
    
                                                                    # omit_inactive = True by default
                                                                    # seed = 0 by default
                                                            
                                                                    ### divide data in in subsets
                                                                    if testRatio==0.0:
                                                                        x_train,x_test,x_tf,y_train,y_test,y_tf = pts.divideWRTtestRatio(X_s,
                                                                                                                                         Y_s,
                                                                                                                                         .5,
                                                                                                                                         train_test_val)
                                                                        x_train=np.append(x_train,x_test,axis=0)
                                                                        y_train=np.append(y_train,y_test,axis=0)
                                                                        x_test = x_tf
                                                                        y_test = y_tf
                                                                    else:
                                                                        x_train,x_test,x_tf,y_train,y_test,y_tf = pts.divideWRTtestRatio(X_s,
                                                                                                                                         Y_s,
                                                                                                                                         testRatio,
                                                                                                                                         train_test_val)
                                                            
                                                                    ### standardize training data
                                                                    x_train, scaler = sc.trainScaler(x_train)
                                                                    x_test = sc.scale(x_test,scaler)
                                                                    
                                                                    for classifier_name in classifier_name_range:
                                                                        uniqueId += 1
                                                            
                                                                        ###
                                                                        print("training started: #",uniqueId)
                                                                        print(classifier_name)
                                                                        
                                                                        if classifier_name in ['svm','linear_svm','random_forest','knn']:
                                                            
                                                                            ### training svm
                                                                            outputDir=svm.classical(x_train,
                                                                                                    y_train,
                                                                                                    x_test,
                                                                                                    y_test,
                                                                                                    rootDirData, 
                                                                                                    classifier_name=classifier_name)
                                                            
                                                                            if testRatio==0.0:                
                                                                                if classifier_name == 'svm':
                                                                                    svm.InputActivationSVM(outputDir,
                                                                                                           DTi,
                                                                                                           deltaT,
                                                                                                           tp,
                                                                                                           [signals_list,target_variable,X_times])
                                                                        else:
                                                                    
                                                                            ### training classifier
                                                                            outputDir = main.callTrainingWOLoading(rootDirLib,rootDirData, #needs to be changed
                                                                                                                   classifier_name=classifier_name,
                                                                                                                   x_train=x_train,
                                                                                                                   y_train=y_train,
                                                                                                                   x_test=x_test,
                                                                                                                   y_test=y_test,
                                                                                                                   nb_epochs=nb_epochs) 
                                                            
                                                                            if testRatio==0.0:                
                                                                                ### input analysis
                                                                                iA.inputAnalysisAndReport(x_train,
                                                                                                          y_train,
                                                                                                          x_test,
                                                                                                          y_test,
                                                                                                          outputDir,
                                                                                                          [signals_list,target_variable,X_times],
                                                                                                          scaler=scaler,
                                                                                                          plot_factor=1,
                                                                                                          zoom_relevant=False)# libRootDir=MLDir)
                                                                                
                                                                        np.save(outputDir+'y_train',y_train)
                                                                        np.save(outputDir+'y_test',y_test)
                                                            
                                                                        print('Save parameters')
                                                                        ### add some results to the parameters.csv file 
                                                                        parameters={"durationOfSeries":durationOfSeries,
                                                                                 "numberOfPrecursors":numberOfPrecursors,
                                                                                 "delayParameterBetweenPrecursorsMEAN":delayParameterBetweenPrecursorsMEAN,
                                                                                 "delayParameterBetweenPrecursorsSTD":delayParameterBetweenPrecursorsSTD,
                                                                                 "meanDurationOfPrecursorsMEAN":meanDurationOfPrecursorsMEAN,
                                                                                 "meanDurationOfPrecursorsSTD":meanDurationOfPrecursorsSTD,
                                                                                 "delayParameterBetweenLastPrecursorAndEventMEAN":\
                                                                                    delayParameterBetweenLastPrecursorAndEventMEAN,
                                                                                 "delayParameterBetweenLastPrecursorAndEventSTD":\
                                                                                    delayParameterBetweenLastPrecursorAndEventSTD,
                                                                                 "meanDurationOfConsequenceMEAN":meanDurationOfConsequenceMEAN,
                                                                                 "meanDurationOfConsequenceSTD":meanDurationOfConsequenceSTD,
                                                                                 "delayParameterBetweenEventAndFirstPrecursorMEAN":\
                                                                                    delayParameterBetweenEventAndFirstPrecursorMEAN,
                                                                                 "delayParameterBetweenEventAndFirstPrecursorSTD":\
                                                                                    delayParameterBetweenEventAndFirstPrecursorSTD,
                                                                                 "delayParameterBetweenEventsMEAN":delayParameterBetweenEventsMEAN,
                                                                                 "delayParameterBetweenEventsSTD":delayParameterBetweenEventsSTD,
                                                                                 "distributionLaw":distributionLaw,
                                                                                 "number_of_noise_channels":number_of_noise_channels,
                                                                                 "seed":seed,
                                                                                 "deltaT":deltaT,
                                                                                 "DTi":DTi,
                                                                                 "DTo":DTo,
                                                                                 "t_p":tp,
                                                                                 "input_variables":input_variables,
                                                                                 "output_variables":output_variables,
                                                                                 "p0_target":p0_target,
                                                                                 "testRatio":testRatio,
                                                                                 "nb_epochs":nb_epochs,
                                                                                 "classifier_name":classifier_name
                                                                                }
                                                                        parameters["testRatio"]=testRatio
                                                                        parameters["p0_target"]=p0_target
                                                                        parameters["classifier_name"]=classifier_name
                                                                        parameters["y_column"]=y_column
                                                                        parameters["sumClass0Test"]=np.size(y_test)-np.sum(y_test)
                                                                        parameters["sumClass0Train"]=np.size(y_train)-np.sum(y_train)
                                                                        parameters["sumClass1Test"]=np.sum(y_test)
                                                                        parameters["sumClass1Train"]=np.sum(y_train)
                                                                        parameters["sizeOfTest"]=np.size(y_test)
                                                                        parameters["sizeOfTrain"]=np.size(y_train)
                                                                        parameters["percentageOfClass1InTest"]=\
                                                                                                                parameters["sumClass1Test"]/parameters["sizeOfTest"]
                                                                        parameters["percentageOfClass0InTest"]=\
                                                                                                                parameters["sumClass0Test"]/parameters["sizeOfTest"]
                                                                        parameters["percentageOfClass1InTrain"]=\
                                                                                                                 parameters["sumClass1Train"]/parameters["sizeOfTrain"]
                                                                        parameters["percentageOfClass0InTrain"]=\
                                                                                                                parameters["sumClass0Train"]/parameters["sizeOfTrain"] 
                                                                        
                                                                        with open(outputDir+'parameters.csv', 'w') as f:
                                                                            for key in parameters.keys():
                                                                                f.write("%s,%s\n"%(key,parameters[key]))
                                                                        
                                                                        with open(outputDir+'X_times.csv', 'w') as f:
                                                                            for X_time in X_times:
                                                                                f.write("%s\n"%X_time)
                                                            
                                                                        with open(outputDir+'alarms_list.csv', 'w') as f:
                                                                            for signal in signals_list:
                                                                                f.write("%s\n"%str(signal))
                                                                            f.write("%s\n"%str(target_variable))
                                                                        
                                                                        print("training finished: #",uniqueId)
