folder with scripts/notebooks to evaluate classification outputs based on activated inputs.

Setup instructions:

1.) no setup required except for installing innvestigate python library

Running instructions:

1.) copy x_train.npy , x_test.npy, y_train.npy, y_test.npy into folder
2.) copy trained fcn model ('best_model.hdf5'; is in the output folder of the model training scripts) into folder
3.) open 'input_activation.ipynb' and run


ToDo:
-adapt codes so that they integrate with our scripts and folder structure
-fix last column in figure