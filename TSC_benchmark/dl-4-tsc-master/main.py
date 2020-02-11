from utils.utils import generate_results_csv
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam

import numpy as np
import sys
import sklearn 
import datetime as dt


def fit_classifier(datasets_dict,dataset_name,classifier_name,output_directory,nb_epochs): 
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64) 
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    if len(x_train.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)

    classifier.fit(x_train,y_train,x_test,y_test, y_true,nb_epochs)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False):
    if classifier_name=='fcn': 
        from classifiers import fcn        
        return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)    
    if classifier_name=='fcn_light': 
        from classifiers import fcn_light        
        return fcn_light.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)    
    if classifier_name=='simple_fcn': 
        from classifiers import simple_fcn        
        return simple_fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='very_simple_fcn': 
        from classifiers import very_simple_fcn        
        return very_simple_fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='fcn_dropout_after_pool': 
        from classifiers import fcn_dropout_after_pool        
        return fcn_dropout_after_pool.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='fcn_dropout_extreme': 
        from classifiers import fcn_dropout_extreme        
        return fcn_dropout_extreme.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='fcn_dropout_twice': 
        from classifiers import fcn_dropout_twice        
        return fcn_dropout_twice.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  classifiers import  mlp 
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='resnet':
        from  classifiers import resnet 
        return resnet.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcnn':
        from  classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory,verbose)
    if classifier_name=='tlenet':
        from  classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory,verbose)
    if classifier_name=='twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory,verbose)
    if classifier_name=='encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='cnn': # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)

############################################### main 

# change this directory for your machine
# it should contain the archive folder containing both univariate and multivariate archives
#root_dir = '/mnt/nfs/casimir/'
#root_dir = '/eos/user/l/lfelsber/LASER/lasML/TSC_benchmark/data'
#root_dir = 'C:/Users/lfelsber/cernbox/LASER/TSC_benchmark/data'

def callTraining(rootDirLib,rootDirData,classifier_name):
    
    # dummy argument
    dataset_name = 'dataset_name'
        
    output_directory = rootDirData+'/results/'+classifier_name+'/'+dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+'/'

    output_directory = create_directory(output_directory)
    
    if output_directory is None: 
        print('Already done')
    else: 

        datasets_dict = read_dataset(rootDirData,dataset_name)
        print(datasets_dict[dataset_name][0].shape)
        print(datasets_dict[dataset_name][1].shape)
        print(datasets_dict[dataset_name][2].shape)
        print(datasets_dict[dataset_name][3].shape)

        fit_classifier(datasets_dict,dataset_name,classifier_name,output_directory)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory+'/DONE')

def callTrainingWOLoading(rootDirLib,rootDirData,classifier_name,
                         x_train, y_train, x_test, y_test, nb_epochs):
    
    # dummy argument
    dataset_name = 'dataset_name'
        
    output_directory = rootDirData+'/results/'+classifier_name+'/'+dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+'/'

    output_directory = create_directory(output_directory)
 
    if output_directory is None: 
        print('Already done')
    else: 

        datasets_dict = {}
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())  
        
        print("shape of x_train",datasets_dict[dataset_name][0].shape)
        print("shape of y_train",datasets_dict[dataset_name][1].shape)
        print("shape of x_test",datasets_dict[dataset_name][2].shape)
        print("shape of y_test",datasets_dict[dataset_name][3].shape)
        
        fit_classifier(datasets_dict,dataset_name,classifier_name,output_directory,nb_epochs)
        
        print('DONE')

        # the creation of this directory means
        create_directory(output_directory+'/DONE')
        
        return output_directory
