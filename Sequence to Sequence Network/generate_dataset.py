"""Generate windowed train-test datasets."""
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


def windowed_dataset(
        capacity_trajectories, min_end_cycle_input=100, input_window=5, output_window=1,
        sample_step_input=1, sample_step_output=1, num_features=1,
        eval_mode=False, end_input_cycle=100
    ):
    """
    Convert the time series dataset into windowed data.

    :param capacity_trajectories: trajectory dataset
    :param min_end_cycle_input: minimum end cycle
    :param input_window: maximum length of the input sequence
    :param output_window: maximum length of the output sequence
    :param sample_step_input: sample step for the input sequence
    :param sample_step_output: sample step for the output sequence
    :param num_features: number of feature for each element of the sequence
    :param eval_mode:
    :param end_input_cycle:
    :return: inputs, outputs       input and output for seq-to-seq model
    """
    input_sample_list, output_sample_list = [], []
    for trajectory in capacity_trajectories:
        max_cycle = trajectory.shape[0]
        max_end_cycle_input = min(
            int(max_cycle * 0.9), max_cycle - sample_step_input - sample_step_output
        ) + 1
        if eval_mode:
            min_end_cycle_input = end_input_cycle
            max_end_cycle_input = end_input_cycle + 1
        for end_cycle in range(min_end_cycle_input, max_end_cycle_input, sample_step_input):
            # there must be at leas one cycle left for the output
            if num_features == 1:
                input_sample = trajectory[0:end_cycle+1:sample_step_input, 1]
                output_sample = trajectory[end_cycle + sample_step_input::sample_step_output, 1]
            elif num_features == 2:
                input_sample = trajectory[0:end_cycle+1:sample_step_input, 0:2]
                output_sample = trajectory[end_cycle + sample_step_input::sample_step_output, 0:2]
            elif num_features == 3:
                input_sample = trajectory[0:end_cycle+1:sample_step_input, :]
                output_sample = trajectory[end_cycle + sample_step_input::sample_step_output, :]
            else:
                raise ValueError(f"{num_features} is not a valid num_features!")
            # Add zero paddings
            input_sample = np.pad(
                input_sample, pad_width=(input_window-len(input_sample), 0), mode='constant'
            )
            output_sample = np.pad(
                output_sample, pad_width=(0, output_window-len(output_sample)), mode='constant'
            )

            # Reshape and append the samples into lists
            input_sample = input_sample.reshape((input_window, 1, num_features))
            output_sample = output_sample.reshape((output_window, 1, num_features))
            input_sample_list.append(input_sample)
            output_sample_list.append(output_sample)

    # Construct the inputs and outputs from the lists
    inputs = np.concatenate(input_sample_list, axis=1)
    outputs = np.concatenate(output_sample_list, axis=1)

    return inputs, outputs
