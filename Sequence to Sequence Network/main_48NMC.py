"""Run S2S-LSTM model on the 48 NMC dataset."""
import sys
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, set_start_method, Queue
from generate_dataset import windowed_dataset, capacity_trajectory_train_test_split
from common_for_different_datasets import prepare_dataloader_eval, ExpertDataSet, task, device_info
from load_data import load_dataset


def prepare_data_train(batch_size, raw_data_path, input_cycle_num, input_window, output_window):
    # load valid cells, initial capacities, and cycle lives
    # load cycle lives for the train and three test datasets
    valid_cells = ["00" + str(i) for i in range(2, 10)] + ["0" + str(i) for i in range(10, 50)]
    initial_capacity = np.genfromtxt(
        raw_data_path + '/initial_capacities.csv', delimiter=',', dtype="float32"
    )
    cycle_lives = np.genfromtxt(
        raw_data_path + '/cycle_lives/cyclelives_80percent.csv', delimiter=',', dtype="float32"
    )

    # load capacity trajectories for the train and three test datasets
    capacity_trajectories = load_dataset(raw_data_path, "Capacity trajectories", valid_cells)

    # split the dataset into training and test datasets
    capacity_trajectories_train, capacity_trajectories_test, idx_train, \
        idx_test = capacity_trajectory_train_test_split(
            capacity_trajectories, split=0.8
        )

    # Split train into training and validation datasets
    np.random.seed(7)
    idx_split = int(0.7 * len(capacity_trajectories_train))
    idx = np.arange(0, len(capacity_trajectories_train))
    np.random.shuffle(idx)
    train_idx = idx[0:idx_split]
    val_idx = idx[idx_split:]
    capacity_trajectories_val = [capacity_trajectories_train[idx] for idx in val_idx]
    capacity_trajectories_training = [capacity_trajectories_train[idx] for idx in train_idx]

    # generate windowed dataset for seq to seq model
    x_train, y_train = windowed_dataset(
        capacity_trajectories_training, min_end_cycle_input=input_cycle_num,
        input_window=input_window, output_window=output_window,
        sample_step_input=5, sample_step_output=50, num_features=1
    )
    x_val, y_val = windowed_dataset(
        capacity_trajectories_val, min_end_cycle_input=input_cycle_num,
        input_window=input_window, output_window=output_window,
        sample_step_input=5, sample_step_output=50, num_features=1
    )
    x_train, y_train, x_val, y_val = torch.from_numpy(x_train).type(torch.Tensor), \
        torch.from_numpy(y_train).type(torch.Tensor), \
        torch.from_numpy(x_val).type(torch.Tensor), \
        torch.from_numpy(y_val).type(torch.Tensor)

    # data format, device info, and data loader
    device, kwargs = device_info()

    train_dataset = ExpertDataSet(x_train, y_train)
    val_dataset = ExpertDataSet(x_val, y_val)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, cycle_lives, idx_train, idx_test, \
           device, capacity_trajectories_train, capacity_trajectories_test, \
            [initial_capacity, initial_capacity]


if __name__ == '__main__':
    dataset = "48 NMC"
    # input cycle number
    start_cycle = 0  # first cycle in the data
    input_cycle = 300  # last cycle in the input data for early prediction
    raw_data_path = '../dataset/48 NMC'
    nominal_cap = 1.85
    input_window = 300
    output_window = 40
    raw_data_path = '/content/drive/My Drive/data for colab/Machine learning for battery early ' \
                    'life prediction/dataset/48 NMC'
    raw_data_path = '../dataset/48 NMC'

    # hyperparams
    early_stopping = True
    num_epoch = 10000
    # layers = [87, 159, 51]
    layers = []
    batch_size = 300
    # lr = 0.0006569517922264813
    lr = 0.0005565684229985987
    gamma = 0.5276248167923726
    # gamma = 0.5372221232058249
    step_start = 72
    step_patience = 31
    act_func = nn.Sigmoid()

    # data preprocess
    train_dataloader, val_dataloader, cycle_lives, idx_train, idx_test, \
        device, capacity_trajectories_train, capacity_trjectories_test, \
        initial_caps_list = prepare_data_train(
            batch_size, raw_data_path, input_cycle - start_cycle, input_window, output_window
        )
    train_dataloader_eval = prepare_dataloader_eval(
        capacity_trajectories_train, input_window=input_window,
        output_window=output_window, end_input_cycle=input_cycle - start_cycle
    )
    tests_dataloader_eval = [
        prepare_dataloader_eval(
            capacity_trjectories_test, input_window=input_window,
            output_window=output_window,
            end_input_cycle=input_cycle - start_cycle
        )
    ]

    # result list
    results = []

    # start multiprocessing
    start_time = time.time()
    set_start_method("spawn")
    num_trials = 3
    training_input_queue = Queue()
    eval_input_queue = Queue()
    training_configs_queue = Queue()
    prediction_output = Queue()

    # model file path for the trial
    result_path = '/content/drive/My Drive/data for colab/Machine learning for battery early life prediction/' \
                  'Sequence to Sequence Network/Trained Models/48 NMC/'
    result_path = 'Trained Models/48 NMC/'

    # prepare shared data for each process
    for trial_idx in range(num_trials):
        training_input_queue.put((trial_idx, train_dataloader, val_dataloader, device, batch_size))
        training_configs_queue.put(
            (
                layers, lr, gamma, act_func, num_epoch, early_stopping, step_start,
                step_patience, raw_data_path, result_path, output_window, input_window,
                start_cycle, input_cycle
            )
        )
        eval_input_queue.put(
            (
                nominal_cap, initial_caps_list, 0.8, train_dataloader_eval, tests_dataloader_eval,
                [cycle_lives, cycle_lives], idx_train, [idx_test]
            )
        )
    # create process
    procs = []
    for trial_idx in range(num_trials):
        proc = Process(
            target=task,
            args=(training_input_queue, training_configs_queue, eval_input_queue, prediction_output)
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    while not prediction_output.empty():
        results.append(prediction_output.get())

    total_time = time.time() - start_time
    print(results)
    print(f"total time is {total_time}!")
