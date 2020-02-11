import warnings
warnings.simplefilter('ignore')

import imp
import numpy as np

from keras.models import load_model

import innvestigate
import innvestigate.utils as iutils

def inputAnalysisAndReport(x_train,y_train,x_test,y_test,
                           output_directory, signals_list, scaler=None, plot_factor=1, libRootDir="../", zoom_relevant=False):

    # Use utility libraries to focus on relevant iNNvestigate routines.
    eutils = imp.load_source("utils", libRootDir+"input_activation/iAutils.py")
    mnistutils = imp.load_source("utils_mnist", libRootDir+"input_activation/utils_mnist.py")


    data=(x_train,y_train,x_test,y_test)

    num_classes = len(np.unique(data[1]))
    label_to_class_name = [str(i) for i in range(num_classes)]
    
    model = load_model(output_directory+'best_model.hdf5')

    
    # Scale to [0, 1] range for plotting.
    def input_postprocessing(X):
        return X


    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

        # Show input
        ("input",                 {},                       input_postprocessing,      "Input"),

        # Function
    #    ("gradient",              {"postprocess": "abs"},   mnistutils.graymap,        "Gradient"),
    #    ("smoothgrad",            {"noise_scale": noise_scale,
    #                               "postprocess": "square"},mnistutils.graymap,        "SmoothGrad"),

        # Signal
    #    ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
    #    ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
    #    ("pattern.net",           {"pattern_type": "relu"}, mnistutils.bk_proj,        "PatternNet"),

        # Interaction
    #    ("pattern.attribution",   {"pattern_type": "relu"}, mnistutils.heatmap,        "PatternAttribution"),
    #    ("deep_taylor.bounded",   {"low": input_range[0],
    #                               "high": input_range[1]}, mnistutils.heatmap,        "DeepTaylor"),
        ("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient"),
    #    ("integrated_gradients",  {"reference_inputs": ri}, mnistutils.heatmap,        "Integrated Gradients"),
    #    ("deep_lift.wrapper",     {"reference_inputs": ri}, mnistutils.heatmap,        "DeepLIFT Wrapper - Rescale"),
    #    ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
    #                                                        mnistutils.heatmap,        "DeepLIFT Wrapper - RevealCancel"),
        ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon")
    ]
    
    
    # Create model without trailing softmax
    try:
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    except:
        print('No softmax detected')
        model_wo_softmax = model
        
    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(data[0], batch_size=256, verbose=1)
        analyzers.append(analyzer)
        
        
    n = 5

    cl1_idx=np.where(y_test==1)[0]
    cl0_idx=np.where(y_test==0)[0]
    
    if len(cl1_idx)>0:
        if len(cl1_idx)>n:
            idx1=np.random.choice(cl1_idx,n,replace=False)
        else:
            idx1=cl1_idx
            print('less than '+str(n)+' active classes in test data')
    else:
        idx1=np.array([]).astype(int)
        print('No active classes in test data!')
        
    if len(cl0_idx)>0:
        if len(cl0_idx)>n:
            idx0=np.random.choice(cl0_idx,n,replace=False)
        else:
            idx0=cl0_idx
            print('less than' +str(n+1)+' active classes in test data')
    else:
        idx0=np.array([]).astype(int)
        print('No inactive classes in test data!')
    
    # try:
    #     try:
    #         idx1=np.random.choice(cl1_idx[0],n,replace=False)
    #     except:
    #         idx1=cl1_idx
    #         print('less than' +str(n)+' active classes in test data')
    # except:
    #     idx1=np.random.randint(0,high=len(y_test),size=n)
    #     print('No active classes in test data!')

    # try:    
    #     idx0=np.random.choice(cl0_idx[0],n,replace=False)
    # except:
    #     idx1=cl1_idx
    #     print('less than' +str(n)+' inactive classes in test data')
        
    test_images_times_idx = np.concatenate((idx0+len(y_train),idx1+len(y_train))) #get right indices to retrieve time intervals from input times   
    test_images = list(zip(data[2][np.concatenate((idx0,idx1))], data[3][np.concatenate((idx0,idx1))]))
    
    # save for later offline analysis
    yt=np.array([el[1] for el in test_images])
    xt=np.array([el[0] for el in test_images])
    
    np.save(output_directory+'yt.npy',yt)
    np.save(output_directory+'xt.npy',xt)



    analysis = np.zeros([len(test_images), len(analyzers), x_train.shape[1],x_train.shape[2]])
    text = []


    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :]

        # Predict final activations, probabilites, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % label_to_class_name[y.astype(int)],    # ground truth label
                     "%.2f" % presm.max(),             # pre-softmax logits
                     "%.2f" % prob.max(),              # probabilistic softmax output  
                     "%s" % label_to_class_name[y_hat] # predicted label
                    ))

        for aidx, analyzer in enumerate(analyzers):
            # Analyze.
            a = analyzer.analyze(x)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = mnistutils.postprocess(a)
            # Store the analysis.
            analysis[i, aidx] = a[0]
            
            
            
    # Prepare the grid as rectengular list
    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    # Plot the analysis.
    eutils.plot_image_grid_w_legend(grid, row_labels_left, row_labels_right, col_labels,
                           file_name=output_directory+"activation.png",
                           signals_list=signals_list, scaler=scaler, plot_factor=plot_factor,
                           zoom_relevant=zoom_relevant,times_idx=test_images_times_idx)
    
    return grid, row_labels_left, row_labels_right, col_labels

def inputAnalysis(x_train,y_train,x_test,y_test,
                 output_directory):

    # Use utility libraries to focus on relevant iNNvestigate routines.
    eutils = imp.load_source("utils", "../input_activation/iAutils.py")
    mnistutils = imp.load_source("utils_mnist", "../input_activation/utils_mnist.py")


    data=(x_train,y_train,x_test,y_test)

    num_classes = len(np.unique(data[1]))
    label_to_class_name = [str(i) for i in range(num_classes)]
    
    model = load_model(output_directory+'best_model.hdf5')

    
    # Scale to [0, 1] range for plotting.
    def input_postprocessing(X):
        return X


    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

        # Show input
        ("input",                 {},                       input_postprocessing,      "Input"),

        # Function
    #    ("gradient",              {"postprocess": "abs"},   mnistutils.graymap,        "Gradient"),
    #    ("smoothgrad",            {"noise_scale": noise_scale,
    #                               "postprocess": "square"},mnistutils.graymap,        "SmoothGrad"),

        # Signal
    #    ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
    #    ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
    #    ("pattern.net",           {"pattern_type": "relu"}, mnistutils.bk_proj,        "PatternNet"),

        # Interaction
    #    ("pattern.attribution",   {"pattern_type": "relu"}, mnistutils.heatmap,        "PatternAttribution"),
    #    ("deep_taylor.bounded",   {"low": input_range[0],
    #                               "high": input_range[1]}, mnistutils.heatmap,        "DeepTaylor"),
        ("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient"),
    #    ("integrated_gradients",  {"reference_inputs": ri}, mnistutils.heatmap,        "Integrated Gradients"),
    #    ("deep_lift.wrapper",     {"reference_inputs": ri}, mnistutils.heatmap,        "DeepLIFT Wrapper - Rescale"),
    #    ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
    #                                                        mnistutils.heatmap,        "DeepLIFT Wrapper - RevealCancel"),
        ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon")
    ]
    
    
    # Create model without trailing softmax
    try:
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    except:
        print('No softmax detected')
        model_wo_softmax = model
        
    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(data[0], batch_size=256, verbose=1)
        analyzers.append(analyzer)
        
        
    n = 5
    try:
        cl1_idx=np.where(y_test==1)
    except:
        cl1_idx=np.zeros(n)
        print('No active classes in test data!')

    cl0_idx=np.where(y_test==0)
    
    try:
        idx1=np.random.choice(cl1_idx[0],n)
    except:
        idx1=np.random.randint(0,high=len(y_test),size=n)
        print('No active classes in test data!')

        
    idx0=np.random.choice(cl0_idx[0],n)

    test_images = list(zip(data[2][np.concatenate((idx0,idx1))], data[3][np.concatenate((idx0,idx1))]))


    analysis = np.zeros([len(test_images), len(analyzers), x_train.shape[1],x_train.shape[2]])
    text = []


    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :]

        # Predict final activations, probabilites, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % label_to_class_name[y],    # ground truth label
                     "%.2f" % presm.max(),             # pre-softmax logits
                     "%.2f" % prob.max(),              # probabilistic softmax output  
                     "%s" % label_to_class_name[y_hat] # predicted label
                    ))

        for aidx, analyzer in enumerate(analyzers):
            # Analyze.
            a = analyzer.analyze(x)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = mnistutils.postprocess(a)
            # Store the analysis.
            analysis[i, aidx] = a[0]
            
            
            
    # Prepare the grid as rectengular list
    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    # Plot the analysis.
    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                           file_name=output_directory+"activation.png")
    
