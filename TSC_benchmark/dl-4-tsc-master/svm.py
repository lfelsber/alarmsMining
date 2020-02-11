# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:20:28 2020

@author: lfelsber
"""
#print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from utils import utils
import main
import datetime as dt
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump


def classical(X_train, y_train, X_test, y_test, rootDirData, classifier_name):
    
    output_directory = rootDirData+'/results/'+classifier_name+'/'+dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")+'/'
                
    output_directory = main.create_directory(output_directory)
    
    orig_shape=X_train.shape

    X_train=X_train.reshape((X_train.shape[0],-1))
    X_test=X_test.reshape((X_test.shape[0],-1))  
    
    # Create a classifier:
    if classifier_name == 'svm':
        classifier = svm.SVC(kernel='linear', tol=1e-5)
        
    if classifier_name == 'linear_svm':
        classifier = LinearSVC(random_state=0, tol=1e-5)
        
    if classifier_name == 'random_forest':
        classifier = RandomForestClassifier(max_features="sqrt" )
        
    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=7)
        
    start_time = time.time() 
    
    # We learn the digits on the first half of the digits
    try:
        classifier.fit(X_train, y_train)
    except:
        print('could not learn a model. aborting')
        return output_directory
    duration = time.time() - start_time
    
    if classifier_name == 'svm':
        try:
            input_relevance = classifier.coef_
            input_relevance = input_relevance.reshape((orig_shape[1],orig_shape[2]))
            
            np.save(output_directory+'input_relevance.npy',input_relevance)
            dump(classifier, output_directory+'svm.joblib') 
            
        except:
            print('Can report input relevance coefficients only for CVM with linear kernel functions.')
    
    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(X_test)
    
    df_metrics=utils.calculate_metrics(y_test, predicted,duration)
    print(output_directory)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)
    
#    classification_report = metrics.classification_report(y_test, predicted)
#    print("Classification report for classifier %s:\n%s\n"
#          % (classifier, classification_report))
#    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
#    disp.figure_.suptitle("Confusion Matrix")
#    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    
    return output_directory

def InputActivationSVM(output_directory, DTi, deltaT, tp, signals_list):
    
    input_relevance = np.load(output_directory+'input_relevance.npy')
    
    plt.figure()
    plt.rcParams['font.family']='serif'
    
    X,Y=np.meshgrid(np.arange(input_relevance.shape[1]+1), np.arange(input_relevance.shape[0]+1))
    plt.pcolormesh(X, Y, input_relevance, cmap='Blues')
    # plt.colorbar(c)
    # plt.imshow(input_relevance, cmap='Blues', interpolation='none')
    plt.xticks(np.arange(len(signals_list[0]))+.5,signals_list[0],rotation=90, fontsize=6)
    deltaT=pd.to_timedelta(deltaT)
    y_labels=[pd.to_timedelta((tp+i)*deltaT.total_seconds(),unit='s') for i in range(DTi+1)]
    plt.yticks(np.arange(DTi+1),y_labels, fontsize=6)
    plt.xlabel('Signals')
    plt.ylabel('Time to predicted event')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.suptitle('Predicting '+str(signals_list[1]),fontsize=4)

    plt.savefig(output_directory+'input_relevance.png', dpi=300)
    
    print('Generated and plotted input relevance.')
    
    
