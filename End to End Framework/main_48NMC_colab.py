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
import argparse
parser = argparse.ArgumentParser()
# Add long and short argument
parser.add_argument("--input_type", type=str, help="set the input type")
# Read arguments from the command line
args = parser.parse_args()

if __name__ == '__main__':
    # base params
    dataset = "48 NMC"
    input_type = args.input_type
    # input_type = "delta QV and dQdV 100-10"
    # input_type = "discharge features"
    if input_type == "delta QV 100-10":
        input_size = 100
    elif input_type == "discharge features":
        input_size = 6
    nominal_cap = 1.85
    raw_data_path = '/content/drive/My Drive/data for colab/Machine learning for battery early life prediction/dataset/' + dataset

    # hyperparameters for model development and training
    empirical_model_name = "exp"
    parama_limit_coeff = -0.2020289411406039
    layers = [174]
    lr = 0.5728854344631988
    ratios = [0.3001574062120874, 0.11235548812908433, 0, 0]
    batch_size = 22
    gamma = 0.8755736416165827
    step_start = 52
    step_patience = 41
    num_epoch = 10000
    loss_func = nn.L1Loss()
    act_func = nn.LeakyReLU()
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
    disk_path = '/content/drive/My Drive/data for colab/Machine learning for battery early life prediction/'
    result_path = disk_path + '/End to End Framework/Trained models/' + dataset + '/'
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



