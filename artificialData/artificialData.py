# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:57:10 2019

@author: tcartier
"""
import numpy as np
import pandas as pd
from datetime import datetime


def computeTimeDelay(delayParameter: float, distributionLaw: str):
    """"
    Function which computes a time delay arrcording to a given distribution law
    exponential: exponential distribution
    deterministic: constant value
    normal: normal distribution which requires two parameters, mean and std
    """
    timeDelay = -1.
    while timeDelay < 0:
        if 'exponential' in distributionLaw:
            timeDelay = np.random.exponential(delayParameter)
        elif 'deterministic' in distributionLaw :
            timeDelay = delayParameter
        elif 'normal' in distributionLaw:
            timeDelay = np.random.normal(delayParameter[0],delayParameter[1])
        else :
            print('Please define an appropriated law')
        
    return timeDelay

def addEvent(FAULT_FAMILY: str,
             FAULT_MEMBER: str,
             FAULT_CODE: str,
             ACTIVE: str,
             SYSTEM_TS: str,
             TSFormat: str,
             PRIORITY: str):
    """
    Function which creates a dictionary and returns it
    """
    event = {"FAULT_FAMILY":FAULT_FAMILY,
             "FAULT_MEMBER":FAULT_MEMBER,
             "FAULT_CODE":FAULT_CODE,
             "ACTIVE":ACTIVE,
             "SYSTEM_TS":str(datetime.fromtimestamp(SYSTEM_TS).strftime(TSFormat)),
             "PRIORITY":PRIORITY}
    return event

def artificialDataCreation2systemsDatetime(durationOfSeries: float,
                           numberOfPrecursors: int,
                           delayParameterBetweenPrecursors: float,
                           meanDurationOfPrecursors: float,
                           delayParameterBetweenLastPrecursorAndEvent: float,
                           meanDurationOfConsequence: float,
                           delayParameterBetweenEventAndFirstPrecursor: float,
                           distributionLaw: str,
                           fileName: str,
                           seed: int,
                          precursorName: str,
                          consequenceName: str):
    """
    Function which generates a given pattern in a time series.
    For each event, a dictionary is appended to a list.
    Finaly the list is of dictionnaries is cast into a pandas dataframe.
    
    Arguments:
    All the time units are in second.
    """
    
    # initiate random generator
    np.random.seed(seed)
    
    # list of parameters
    precursorFamily = "family_"+precursorName
    precursorMember = "member_"+precursorName
    precursorCode   = "code_"+precursorName
    consequenceFamily = "family_"+consequenceName
    consequenceMember = "member_"+consequenceName
    consequenceCode   = "code_"+consequenceName   

    # date format like '2012-05-01 00:00:00'
    TSFormat = "%Y-%m-%d %H:%M:%S.%f" #"%d-%b-%y %I.%M.%S.%f %p"     

    
    data = [] # empty list we fill with events
    
    # time = datetime.timestamp(datetime.now())    
    time = datetime.timestamp(datetime(2015, 3, 3))
    endOfTime = time + durationOfSeries
    
    # loop on the total number of patterns
    while time < endOfTime:
            
        # loop on the number of triggering Occurences
        for idPrecursor in range(0,numberOfPrecursors):
            
            # rising precursor alarm
            time = time + computeTimeDelay(delayParameterBetweenPrecursors, distributionLaw)
            if time < endOfTime:
                data.append(addEvent(precursorFamily,
                                     precursorMember,
                                     precursorCode,
                                     "Y",
                                     time,
                                     TSFormat,
                                     2))
            else:
                break

            # clearing precursor alarm
            time = time + computeTimeDelay(meanDurationOfPrecursors, distributionLaw)
            if time < endOfTime:
                data.append(addEvent(precursorFamily,
                                     precursorMember,
                                     precursorCode,
                                     "N",
                                     time,
                                     TSFormat,
                                     2))                          
            else:
                break
                
        # rising consequence alarm
        time = time + computeTimeDelay(delayParameterBetweenLastPrecursorAndEvent, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(consequenceFamily,
                                 consequenceMember,
                                 consequenceCode,
                                 "Y",
                                 time,
                                 TSFormat,
                                 3))      
        else:
            break
    
        # clearing consequence alarm
        time = time + computeTimeDelay(meanDurationOfConsequence, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(consequenceFamily,
                                 consequenceMember,
                                 consequenceCode,
                                 "N",
                                 time,
                                 TSFormat,
                                 3))
        else:
            break
            
        # adding some delay between the last consequence and the next precursor
        time = time + computeTimeDelay(delayParameterBetweenEventAndFirstPrecursor, distributionLaw)

    df = pd.DataFrame.from_dict(data)
    if fileName: # only write data if the fileName is not "None"
        df.to_csv(fileName,index=False)
        
    return df

def artificialDataCreation1systemDatetime(durationOfSeries: float,
                           delayParameterBetweenEvents: float,
                           meanDurationOfEvents: float,
                           distributionLaw: str,
                           fileName: str,
                           seed: int,
                          eventName: str,
                          alarmLevel: int):
    """
    Function which generates a random distribution of a given event.
    
    Arguments:
    All the time units are in second.
    """
    
    # initiate random generator
    np.random.seed(seed)
    
    # list of parameters
    eventFamily = "family_"+eventName
    eventMember = "member_"+eventName
    eventCode   = "code_"+eventName
    
    # date format like '2012-05-01 00:00:00'
    TSFormat = "%Y-%m-%d %H:%M:%S.%f" #"%d-%b-%y %I.%M.%S.%f %p" 
    
    data = [] # empty list we fill with events
    
    # time = datetime.timestamp(datetime.now())    
    time = datetime.timestamp(datetime(2015, 3, 3))
    endOfTime = time + durationOfSeries
    
    # loop on the total number of patterns
    while time < endOfTime:
        
        # rising precursor alarm
        time = time + computeTimeDelay(delayParameterBetweenEvents, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(eventFamily,
                                 eventMember,
                                 eventCode,
                                 "Y",
                                 time,
                                 TSFormat,
                                 alarmLevel))
        else:
            break

        # clearing precursor alarm
        time = time + computeTimeDelay(meanDurationOfEvents, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(eventFamily,
                                 eventMember,
                                 eventCode,
                                 "N",
                                 time,
                                 TSFormat,
                                 alarmLevel))                          
        else:
            break

    df = pd.DataFrame.from_dict(data)
    if fileName: # only write data if the fileName is not "None"
        df.to_csv(fileName,index=False)
        
    return df


def artificialDataCreation2systems(durationOfSeries: float,
                           numberOfPrecursors: int,
                           delayParameterBetweenPrecursors: float,
                           meanDurationOfPrecursors: float,
                           delayParameterBetweenLastPrecursorAndEvent: float,
                           meanDurationOfConsequence: float,
                           delayParameterBetweenEventAndFirstPrecursor: float,
                           distributionLaw: str,
                           fileName: str,
                           seed: int,
                          precursorName: str,
                          consequenceName: str):
    """
    Function which generates a given pattern in a time series.
    For each event, a dictionary is appended to a list.
    Finaly the list is of dictionnaries is cast into a pandas dataframe.
    
    Arguments:
    All the time units are in second.
    """
    
    # initiate random generator
    np.random.seed(seed)
    
    # list of parameters
    precursorFamily = "family_"+precursorName
    precursorMember = "member_"+precursorName
    precursorCode   = "code_"+precursorName
    consequenceFamily = "family_"+consequenceName
    consequenceMember = "member_"+consequenceName
    consequenceCode   = "code_"+consequenceName   

    
    # date format of LASER
    TSFormat = "%d-%b-%y %I.%M.%S.%f %p" 
    
    data = [] # empty list we fill with events
    
    time = datetime.timestamp(datetime.now())    
    endOfTime = time + durationOfSeries
    
    # loop on the total number of patterns
    while time < endOfTime:
            
        # loop on the number of triggering Occurences
        for idPrecursor in range(0,numberOfPrecursors):
            
            # rising precursor alarm
            time = time + computeTimeDelay(delayParameterBetweenPrecursors, distributionLaw)
            if time < endOfTime:
                data.append(addEvent(precursorFamily,
                                     precursorMember,
                                     precursorCode,
                                     "Y",
                                     time,
                                     TSFormat,
                                     2))
            else:
                break

            # clearing precursor alarm
            time = time + computeTimeDelay(meanDurationOfPrecursors, distributionLaw)
            if time < endOfTime:
                data.append(addEvent(precursorFamily,
                                     precursorMember,
                                     precursorCode,
                                     "N",
                                     time,
                                     TSFormat,
                                     2))                          
            else:
                break
                
        # rising consequence alarm
        time = time + computeTimeDelay(delayParameterBetweenLastPrecursorAndEvent, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(consequenceFamily,
                                 consequenceMember,
                                 consequenceCode,
                                 "Y",
                                 time,
                                 TSFormat,
                                 3))      
        else:
            break
    
        # clearing consequence alarm
        time = time + computeTimeDelay(meanDurationOfConsequence, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(consequenceFamily,
                                 consequenceMember,
                                 consequenceCode,
                                 "N",
                                 time,
                                 TSFormat,
                                 3))
        else:
            break
            
        # adding some delay between the last consequence and the next precursor
        time = time + computeTimeDelay(delayParameterBetweenEventAndFirstPrecursor, distributionLaw)

    df = pd.DataFrame.from_dict(data)
    if fileName: # only write data if the fileName is not "None"
        df.to_csv(fileName,index=False)
        
    return df

def artificialDataCreation1system(durationOfSeries: float,
                           delayParameterBetweenEvents: float,
                           meanDurationOfEvents: float,
                           distributionLaw: str,
                           fileName: str,
                           seed: int,
                          eventName: str,
                          alarmLevel: int):
    """
    Function which generates a random distribution of a given event.
    
    Arguments:
    All the time units are in second.
    """
    
    # initiate random generator
    np.random.seed(seed)
    
    # list of parameters
    eventFamily = "family_"+eventName
    eventMember = "member_"+eventName
    eventCode   = "code_"+eventName
    
    # date format of LASER
    TSFormat = "%d-%b-%y %I.%M.%S.%f %p" 
    
    data = [] # empty list we fill with events
    
    time = datetime.timestamp(datetime.now())    
    endOfTime = time + durationOfSeries
    
    # loop on the total number of patterns
    while time < endOfTime:
        
        # rising precursor alarm
        time = time + computeTimeDelay(delayParameterBetweenEvents, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(eventFamily,
                                 eventMember,
                                 eventCode,
                                 "Y",
                                 time,
                                 TSFormat,
                                 alarmLevel))
        else:
            break

        # clearing precursor alarm
        time = time + computeTimeDelay(meanDurationOfEvents, distributionLaw)
        if time < endOfTime:
            data.append(addEvent(eventFamily,
                                 eventMember,
                                 eventCode,
                                 "N",
                                 time,
                                 TSFormat,
                                 alarmLevel))                          
        else:
            break

    df = pd.DataFrame.from_dict(data)
    if fileName: # only write data if the fileName is not "None"
        df.to_csv(fileName,index=False)
        
    return df
