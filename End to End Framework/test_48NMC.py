import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import time
from data_preprocessing import prepare_data_train
import torch
from common import task, device_info
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, set_start_method, Queue
from torch import nn


if __name__ == '__main__':
    # base params
    dataset = "48 NMC"
    # input_type = "delta QV 100-10"
    # input_type = "delta QV and dQdV 100-10"
    input_type = "discharge features"
    if input_type == "delta QV 100-10":
        input_size = 100
    elif input_type == "discharge features":
        input_size = 6
    nominal_cap = 1.85
    raw_data_path = '../dataset/48 NMC'

    # hyperparameters for model development and training
    empirical_model_name = "exp_linear2"
    parama_limit_coeff = -0.20591978775691885
    layers = [140, 81]
    lr = 0.9928478250018733
    ratios = [0.41108381916490033, 0.9758581581916301, 0, 0]
    batch_size = 34
    gamma = 0.825804307585216
    step_start = 52
    step_patience = 46
    num_epoch = 10000
    loss_func = nn.L1Loss()
    act_func = nn.ReLU()
    number_nn = 5
    early_stopping = True

    # data preprocess
    precision = "float32"
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    training_dataloader, val_dataloader, device, train_dataset, test_datast_list = prepare_data_train(
        dataset, input_type, raw_data_path, precision, batch_size=batch_size)

    # create eval dataloaders, not shuffle!!!
    device, kwargs = device_info()
    batch_size = 64
    dataloader_eval_list = [DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, **kwargs)]
    for test_dataset in test_datast_list:
        dataloader_eval_list.append(DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs))

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
    raw_data_path = disk_path + project_path + '/dataset/' + dataset
    result_path = disk_path + project_path + '/End to End Framework/Trained models/' + dataset + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for trial_idx in range(num_trials):
        training_input_queue.put((trial_idx, training_dataloader, val_dataloader, device))
        eval_input_queue.put((nominal_cap, 0.8, dataloader_eval_list))
        training_configs.put((empirical_model_name, parama_limit_coeff, input_size, number_nn, layers, lr, gamma, act_func, ratios,
                              loss_func, num_epoch, early_stopping, step_start, step_patience, raw_data_path,
                              result_path))

    # create process
    procs = []
    for trial_idx in range(num_trials):
        proc = Process(target=task, args=(training_input_queue, training_configs, eval_input_queue, prediction_output))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    while not prediction_output.empty():
        results.append(prediction_output.get())

    total_time = time.time() - start_time
    print(results)
    print(f"total time is {total_time}!")



