"""Apply end-to-end framework to 169 LFP dataset."""
import time
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_preprocessing import prepare_data_train
import torch
from common import task, device_info
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, set_start_method, Queue
from torch import nn


if __name__ == '__main__':
    # base params
    DATASET = "169 LFP"
    # input_type = "delta QV 100-10"
    # input_type = "delta QV and dQdV 100-10"
    INPUT_TYPE = "discharge features"
    if INPUT_TYPE == "delta QV 100-10":
        INPUT_SIZE = 100
    elif INPUT_TYPE == "discharge features":
        INPUT_SIZE = 6
    NOMINAL_CAP = 1.1
    RAW_DATA_PATH = '../dataset/169 LFP'

    # hyperparameters for model development and training
    empirical_model_name = "power_law1"
    parama_limit_coeff = -0.38048296535227183
    layers = [188]
    lr = 0.738452038475755
    ratios = [0.6881568330689313, 0.9281677574952851, 0, 0]
    batch_size = 18
    gamma = 0.8588431302153151
    step_start = 72
    step_patience = 46
    num_epoch = 10000
    loss_func = nn.L1Loss()
    act_func = nn.Sigmoid()
    number_nn = 5
    early_stopping = True

    # data preprocess
    precision = "float32"
    torch.set_default_dtype(torch.float32)
    training_dataloader, val_dataloader, device, \
        train_dataset, test_datast_list = prepare_data_train(
            DATASET, INPUT_TYPE, RAW_DATA_PATH, precision, batch_size=batch_size
        )

    # create eval dataloaders, not shuffle!!!
    device, kwargs = device_info()
    batch_size = 64
    dataloader_eval_list = [
        DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            shuffle=False, **kwargs
        )
    ]
    for test_dataset in test_datast_list:
        dataloader_eval_list.append(
            DataLoader(
                dataset=test_dataset, batch_size=batch_size,
                shuffle=False, **kwargs
            )
        )

    # result list fed into a notebook for metric summary
    results = []

    # start multiprocessing
    num_trials = 10
    start_time = time.time()
    set_start_method("spawn")
    training_input_queue = Queue()
    eval_input_queue = Queue()
    prediction_output = Queue()
    training_configs =  Queue()

    # model file path for the trial
    disk_path = "C:/Users/jinqiang/Dropbox/IEC Project Materials/Code"
    project_path = "/battery_early_capacity_trajectory_prediction"
    raw_data_path = disk_path + project_path + '/dataset/' + DATASET
    result_path = disk_path + project_path + '/End to End Framework/Trained models/' + DATASET + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for trial_idx in range(num_trials):
        training_input_queue.put((trial_idx, training_dataloader, val_dataloader, device))
        eval_input_queue.put((NOMINAL_CAP, 0.8, dataloader_eval_list))
        training_configs.put(
            (
                empirical_model_name, parama_limit_coeff, INPUT_SIZE,
                number_nn, layers, lr, gamma, act_func, ratios,
                loss_func, num_epoch, early_stopping, step_start,
                step_patience, raw_data_path, result_path
            )
        )

    # create process
    procs = []
    for trial_idx in range(num_trials):
        proc = Process(
            target=task,
            args=(
                training_input_queue, training_configs,
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
