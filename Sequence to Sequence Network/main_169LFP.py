"""Run S2S-LSTM model on the 169 LFP dataset."""
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, set_start_method, Queue
from generate_dataset import windowed_dataset
from common_for_different_datasets import prepare_dataloader_eval, ExpertDataSet, task, device_info
from load_data import load_dataset

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def normalize_capacity_trajectory(capacity_trjectories):
    """Normalize the capacity trajectory."""
    inital_caps = np.full((len(capacity_trjectories),), np.nan)
    for idx, _ in enumerate(capacity_trjectories):
        inital_caps[idx] = capacity_trjectories[idx][0, 2]
        capacity_trjectories[idx][:, 1] = capacity_trjectories[idx][:, 1] / inital_caps[idx]
    return inital_caps

def prepare_data_train(batch_size, raw_data_path, input_cycle_num):
    """Prepare training dataloaders."""
    # load valid cells, initial capacities, and cycle lives
    # load cycle lives for the train and three test datasets
    cycle_lives_train = np.genfromtxt(raw_data_path + '/cycle_lives/train_cycle_lives.csv',
                                      delimiter=',', dtype="float32")
    cycle_lives_test1 = np.genfromtxt(raw_data_path + '/cycle_lives/test1_cycle_lives.csv',
                                      delimiter=',', dtype="float32")
    cycle_lives_test2 = np.genfromtxt(raw_data_path + '/cycle_lives/test2_cycle_lives.csv',
                                      delimiter=',', dtype="float32")
    cycle_lives_test3 = np.genfromtxt(raw_data_path + '/cycle_lives/test3_cycle_lives.csv',
                                      delimiter=',', dtype="float32")

    # load capacity trajectories for the train and three test datasets
    valid_cells_train = ["cell" + str(i) for i in range(1, 42)]
    valid_cells_test1 = ["cell" + str(i) for i in range(1, 43)]
    valid_cells_test2 = ["cell" + str(i) for i in range(1, 41)]
    valid_cells_test3 = ["cell" + str(i) for i in range(1, 46)]
    capacity_trajectories_train = load_dataset(
        raw_data_path, "discharge_capacity/train", valid_cells_train
    )
    capacity_trajectories_test1 = load_dataset(
        raw_data_path, "discharge_capacity/test1", valid_cells_test1
    )
    capacity_trajectories_test2 = load_dataset(
        raw_data_path, "discharge_capacity/test2", valid_cells_test2
    )
    capacity_trajectories_test3 = load_dataset(
        raw_data_path, "discharge_capacity/test3", valid_cells_test3
    )

    # Normalize the capacity trajectory
    inital_caps_train = normalize_capacity_trajectory(capacity_trajectories_train)
    inital_caps_test1 = normalize_capacity_trajectory(capacity_trajectories_test1)
    inital_caps_test2 = normalize_capacity_trajectory(capacity_trajectories_test2)
    inital_caps_test3 = normalize_capacity_trajectory(capacity_trajectories_test3)

    # Split Xtrain and Y_train into training and validation datasets
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
        capacity_trajectories_training, min_end_cycle_input=input_cycle_num, input_window=500,
        output_window=50, sample_step_input=5, sample_step_output=50, num_features=1
    )
    x_val, y_val = windowed_dataset(
        capacity_trajectories_val, min_end_cycle_input=input_cycle_num, input_window=500,
        output_window=50, sample_step_input=5, sample_step_output=50, num_features=1
    )
    x_train, y_train, x_val, y_val = torch.from_numpy(x_train).type(torch.Tensor), \
        torch.from_numpy(y_train).type(torch.Tensor), \
        torch.from_numpy(x_val).type(torch.Tensor), \
        torch.from_numpy(y_val).type(torch.Tensor)

    device, kwargs = device_info()
    train_dataset = ExpertDataSet(x_train, y_train)
    val_dataset = ExpertDataSet(x_val, y_val)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_dataloader, val_dataloader, \
        [cycle_lives_train, cycle_lives_test1, cycle_lives_test2, cycle_lives_test3], \
        device, capacity_trajectories_train, \
        [capacity_trajectories_test1, capacity_trajectories_test2, capacity_trajectories_test3],\
        [inital_caps_train, inital_caps_test1, inital_caps_test2, inital_caps_test3]


if __name__ == '__main__':
    dataset = "169 LFP"
    # input cycle number
    start_cycle = 2  # first cycle in the data
    input_cycle = 300  # last cycle in the input data for early prediction
    raw_data_path = '../dataset/169 LFP'
    nominal_cap = 1.1
    input_window = 500
    output_window = 50
    raw_data_path = '/content/drive/My Drive/data for colab/Machine learning for battery early ' \
                    'life prediction/dataset/169 LFP'
    raw_data_path = '../dataset/169 LFP'
    # hyperparams
    early_stopping = True
    num_epoch = 10000
    layers = [87, 159, 51]
    batch_size = 110
    lr = 0.0006569517922264813
    gamma = 0.5372221232058249
    step_start = 82
    step_patience = 21
    act_func = nn.Tanh()

    # data preprocess
    train_dataloader, val_dataloader, cycle_lives_all, device, capacity_trajectories_train, \
    capacity_trajectories_tests, inital_caps_list = prepare_data_train(batch_size, raw_data_path,
                                                                       input_cycle - start_cycle)

    train_dataloader_eval = prepare_dataloader_eval(
        capacity_trajectories_train, input_window=input_window,
        output_window=output_window, end_input_cycle=input_cycle - start_cycle
    )
    tests_dataloader_eval = [
        prepare_dataloader_eval(
            capacity_trajectories_tests[0], input_window=input_window,
            output_window=output_window, end_input_cycle=input_cycle - start_cycle
        ),
        prepare_dataloader_eval(
            capacity_trajectories_tests[1], input_window=input_window,
            output_window=output_window, end_input_cycle=input_cycle - start_cycle
        ),
        prepare_dataloader_eval(
            capacity_trajectories_tests[2], input_window=input_window,
            output_window=output_window, end_input_cycle=input_cycle - start_cycle
        )
    ]
    nominal_cap = 1.1

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
    idx_train = np.arange(0, len(cycle_lives_all[0]))
    idx_test_list = [np.arange(0, len(cycle_lives_all[test_id])) for test_id in range(1, 4)]
    # model file path for the trial
    result_path = '/content/drive/My Drive/data for colab/Machine learning for battery early life prediction/' \
                  'Sequence to Sequence Network/Trained Models/169 LFP/'
    result_path = 'Trained Models/169 LFP/'

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
                nominal_cap, inital_caps_list, 0.8, train_dataloader_eval, tests_dataloader_eval,
                cycle_lives_all, idx_train, idx_test_list
            )
        )
    # create process
    procs = []
    for trial_idx in range(num_trials):
        proc = Process(
            target=task,
            args=(
                training_input_queue, training_configs_queue,
                eval_input_queue, prediction_output
            )
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
