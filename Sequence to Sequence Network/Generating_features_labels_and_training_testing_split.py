######################################################################
#
# Script to generate the training and testing
# arrays seperated into features and labels.
# Authors: Neil Sengupta, Weihan Li
#
######################################################################
# Comments:
#
# This script uses the 'mat4py', 'h5py', and 'sklearn' packages
# which needs to be installed prior to usage
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import mat4py as mpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import h5py as st

def array_shuffler(array_1, array_2):
    # function to shuffle two arrays of equal length simultaneously
    # while maintaining correlation between individual rows
    assert len(array_1) == len(array_2)  # confirm equal lengths
    # generate random permutation of rows up to array size
    p = np.random.permutation(len(array_1))
    # slice both arrays with the same permutation matrix
    return array_1[p], array_2[p]

def generate_dataset_from_rawfile(datapath, dataname, savepath, shuffle=True):
    # function to take the "test/train.mat" files generated by the MATLAB script
    # and split the structure into feature arrays and label arrays
    # with an option to shuffle array rows

    # datapath = The full file path of the .mat file to be used as input
    # dataname = The struct variable name present in the .mat file
    # savepath = The full file path of the location for saving the arrays
    # shuffle  = If true, shuffles the feature and label array simultaneously

    # temporary dictionary from .mat file
    data_loader = mpy.loadmat(datapath)[dataname]
    # Generate pandas data frame 'data' from the .mat file
    data = pd.DataFrame.from_dict(data_loader)

    number_of_rows = len(data)                                  # number of rows of the data frame
    number_of_samples = 0                                       # variable for number of samples in each input row
    number_of_targets = 0                                       # variable for number of samples in each target row
    inputlist = []                                              # initiate input array as list
    targetlist = []                                             # initiate target array as list

    # Extract each row from the data frame
    for row in range(number_of_rows):
        input_row = data.iloc[row, -2]                          # extract capacity history as input
        number_of_samples = len(input_row)                      # number of samples in the current input row
        current_input_row = np.zeros((number_of_samples, 1))    # initiate current input row as numpy array
        for sample in range(number_of_samples):
            # populate current row with sample values from capacity history
            current_input_row[sample, 0] = input_row[sample]
        inputlist.append(current_input_row)                     # append current row to input list

        target_row = data.iloc[row, -1]                         # extract future capacity degradation curve as output
        number_of_targets = len(target_row)
        targetlist.append(target_row)                           # append current target row to target list

    # convert input and target lists to numpy arrays and assert proper dimensions
    input_array = np.asarray(inputlist).reshape((number_of_rows, number_of_samples, 1))
    target_array = np.asarray(targetlist).reshape((number_of_rows, number_of_targets, 1))

    if shuffle:
        # shuffle input and target arrays in tandem if shuffle = true in function parameters
        features_file, labels_file = array_shuffler(input_array, target_array)
        print('Shuffled set', end=' - ')
    else:
        # If shuffle = false, then just pass the original arrays
        print('Non Shuffled set', end=' - ')
        features_file, labels_file = input_array, target_array

    print(features_file.shape, labels_file.shape)             # confirm proper dimensions of the input and target files

    # save the features and labels in appropriate locations
    featurespath = savepath + 'features.h5'
    labelspath = savepath + 'labels.h5'
    with st.File(featurespath, 'w') as storedata:
        storedata.create_dataset("features", data=features_file)
    with st.File(labelspath, 'w') as storedata:
        storedata.create_dataset("labels", data=labels_file)

    return features_file, labels_file


filepath_of_datasets = ''                            # Enter full file path of the .mat files of the dataset
file_matlab_nametag = 'S2SLearning_14-AUG-2022_'       # Enter the version tag of the file generated in MATLAB
output_nametag = 'Lifetime_Prediction_Benchmark'             # Enter the output tag of the files to be saved
output_datetag = '14_AUG_2020'                               # Enter the date as part of the file name to be saved
training_structure_variable_name = 'Train_Set'               # Variable name of the training set in the .mat structure
testing_structure_variable_name = 'Test_Set'                 # Variable name of the testing set in the .mat structure

# generate the full file path and names of the .mat files to be used as input to the functions
matlab_trainingset_filepath = filepath_of_datasets + file_matlab_nametag + training_structure_variable_name + '.mat'
matlab_testingset_filepath = filepath_of_datasets + file_matlab_nametag + testing_structure_variable_name + '.mat'

# generate the full file path for saving the features and labels arrays from the functions
training_arrays_save_path = filepath_of_datasets + output_nametag + output_datetag + '_Train_'
testing_arrays_save_path = filepath_of_datasets + output_nametag + output_datetag + '_Test_'

# call the functions to generate the arrays
X_Data, y_Data = generate_dataset_from_rawfile(matlab_trainingset_filepath, training_structure_variable_name, training_arrays_save_path, shuffle=True)
X_Test, y_Test = generate_dataset_from_rawfile(matlab_testingset_filepath, testing_structure_variable_name, testing_arrays_save_path, shuffle=False)

# split the training array further for training and validation while training each epoch
# This part can be ignored if training-validation split is done directly inside the model
training_validation_fraction = 0.15            # fraction of samples to be used as validation while training
# The above variable is a percentage to fraction, and thus must be between 0 to 1

X_train, X_val, y_train, y_val = tts(X_Data, y_Data, test_size=training_validation_fraction, shuffle=True)
print('--- shuffled split as: ', X_train.shape[0], ' samples, and testing: ', X_val.shape[0], 'samples ---')


######################################################################
# Further code for model creation and training
######################################################################