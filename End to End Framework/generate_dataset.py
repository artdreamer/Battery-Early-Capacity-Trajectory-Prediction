"""Split the dataset."""
import numpy as np

def capacity_trajectory_train_test_split(data, split=0.8, shuffle=True):
    """
    Split the capacity trajectory dataset into training and test dataset.

    :param data: the entire dataset including features and labels
    :param split: proportion of the data for training dataset
    :param shuffle: whether shuffle the data for the split
    :return:
    """
    np.random.seed(0)
    idx_split = int(split * len(data))
    idx = np.arange(0, len(data))
    if shuffle:
        np.random.shuffle(idx)
    idx_train = idx[0:idx_split]
    idx_test = idx[idx_split:]

    data_train, data_test = [], []
    for idx in idx_train:
        data_train.append(data[idx])

    for idx in idx_test:
        data_test.append(data[idx])

    return data_train, data_test, idx_train, idx_test
