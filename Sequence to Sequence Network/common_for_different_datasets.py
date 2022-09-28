"""Common classes and funcion for running S2S models."""
import time
import os
from scipy import interpolate
from scipy.optimize import fsolve, root_scalar
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from generate_dataset import windowed_dataset
from seq_to_seq_modules import SeqToSeqLSTM
import matplotlib.pyplot as plt


class ExpertDataSet(Dataset):
    """Format the dataset."""

    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs
        self.input.share_memory_()
        self.output.share_memory_()

    def __getitem__(self, index):
        return self.input[:, index, :], self.output[:, index, :]

    def __len__(self):
        return self.input.shape[1]



def interpolated_capacity_cycle_func(
        x, capacity_trajectories_predicted, cut_off_capacity, nominal_cap, inital_cap
    ):
    """Wrap the eval of interpolated func."""
    return interpolate.splev(x, capacity_trajectories_predicted, der=0) -\
        cut_off_capacity * nominal_cap / inital_cap

def fprime(x, capacity_trajectories_predicted, cut_off_capacity, nominal_cap, inital_cap):
    """Get the first-order derivative."""
    return interpolate.splev(x, capacity_trajectories_predicted, der=1)

def device_info():
    """Get device info."""
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    return device, kwargs



def prepare_dataloader_eval(
        capacity_trajectories, input_window=300, output_window=40, end_input_cycle=100
    ):
    """Prepare a dataloader for evaluation, i.e., batch_size = 1, first 100 cycle input."""
    # generate windowed dataset for seq to seq model for eval
    x, y = windowed_dataset(
        capacity_trajectories, input_window=input_window, output_window=output_window,
        sample_step_input=5, sample_step_output=50, num_features=1, eval_mode=True,
        end_input_cycle=end_input_cycle
    )
    x, y = torch.from_numpy(x).type(torch.Tensor), torch.from_numpy(y).type(torch.Tensor)

    # data format, device info, and data loader
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(1)
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    dataset = ExpertDataSet(x, y)
    dataloader_eval = DataLoader(dataset=dataset, batch_size=1, shuffle=False, **kwargs)
    return dataloader_eval


def cycle_life_metric(
        cut_off_capacity, capacity_trajectories_predicted, cycle_lives, idx_eval, trial_idx,
        nominal_cap, initial_caps, last_cycle_pred_train
    ):
    """Calculate cycle life predictive metrics."""
    cycle_lives_pred = np.full(len(cycle_lives), np.nan)
    for idx, cell_idx in enumerate(idx_eval):
        cycle_lives_pred[cell_idx] = root_scalar(
            interpolated_capacity_cycle_func,
            args=(
                capacity_trajectories_predicted[idx],
                cut_off_capacity, nominal_cap, initial_caps[cell_idx]
            ),
            bracket=(100, last_cycle_pred_train[idx])
        ).root
    mae_test = np.abs(cycle_lives_pred[idx_eval] - cycle_lives[idx_eval]).mean()
    mape_test = np.abs(
        (cycle_lives_pred[idx_eval] - cycle_lives[idx_eval])/cycle_lives[idx_eval]
    ).mean() * 100
    rmse_test = np.sqrt(((cycle_lives_pred[idx_eval] - cycle_lives[idx_eval])**2).mean())
    print(f"#{trial_idx} trial: Cycle life predictive MAE is {mae_test} cycles, RMSE is {rmse_test} cycles")
    return mae_test, rmse_test, mape_test



def task(training_input_queue, training_configs_queue, eval_input_queue, prediction_output):
    """A single task function defined for multiprocessing."""
    trial_idx, train_dataloader, val_dataloader, device, batch_size = training_input_queue.get()
    nominal_cap, initial_caps_list, cut_off_capacity, train_dataloader_eval, \
        tests_dataloader_eval, cycle_lives_all, idx_train, \
        idx_test_list = eval_input_queue.get()
    layers, lr, gamma, act_func, num_epoch, early_stopping, step_start, \
        step_patience, raw_data_path, result_path, target_len, input_window, \
        start_cycle, end_input_cycle = training_configs_queue.get()
    monitor_path = result_path + f'{trial_idx}'
    if not os.path.exists(monitor_path):
        os.mkdir(monitor_path)

    # # Step 1 train the model
    # Step 1 train the model
    seq_to_seq_model = SeqToSeqLSTM(input_size=1, hidden_size=100, num_layers=4, device=device,
                                    dense_layers=layers, act_func=act_func).to(device)
    start_time = time.time()
    _, _ = seq_to_seq_model.train_encoder_decoder(
        train_dataloader, val_dataloader, n_epochs=num_epoch,
        target_len=target_len, batch_size=batch_size,
        training_prediction='mixed_teacher_forcing',
        teacher_forcing_ratio=0.6, learning_rate=lr, gamma=gamma, dynamic_tf=False,
        early_stopping=early_stopping, step_start=step_start, step_patience=step_patience,
        monitor_path=monitor_path,
    )
    training_time = time.time() - start_time

    # Step 2 Evaluate predicted capacity trajectories
    seq_to_seq_model.load_state_dict(
        torch.load(
            monitor_path + "/best_model.pt", map_location=torch.device('cpu')
        )
    )
    seq_to_seq_model.eval()

    start_time = time.time()
    # first evaluate training performance: capacity trajectory and cycle life prediction
    print("training performance")
    if not os.path.exists(result_path + f"Figures/trail{trial_idx}"):
        os.mkdir(result_path + f"Figures/trail{trial_idx}")
    capacity_trajectories_predicted_train, cap_mae_train, cap_rmse_train, \
        cap_mape_train, last_cycle_pred_train = evaluate_dataset(
            seq_to_seq_model, train_dataloader_eval, target_len, cycle_lives_all[0],
            idx_train, input_window=input_window, start_cycle=start_cycle,
            end_input_cycle=end_input_cycle, show_plot=False,
            work_path=result_path + f"Figures/trail{trial_idx}/predicted_trajectories_train.png",
            initial_caps = initial_caps_list[0],
        )
    life_mae_train, life_rmse_train, life_mape_train = cycle_life_metric(
        cut_off_capacity, capacity_trajectories_predicted_train, cycle_lives_all[0],
        idx_train, trial_idx, nominal_cap, initial_caps_list[0], last_cycle_pred_train
    )

    # create metric lists
    cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape, \
        last_cycle_pred_list = [], [], [], [], [], [], []
    cap_rmse.append(cap_rmse_train)
    cap_mae.append(cap_mae_train)
    cap_mape.append(cap_mape_train)
    life_rmse.append(life_rmse_train)
    life_mae.append(life_mae_train)
    life_mape.append(life_mape_train)
    last_cycle_pred_list.append(last_cycle_pred_train)

    # second evaluate test performance: capacity trajectory and cycle life prediction
    print("test performance")
    test_id = 1
    for cycle_lives_test, test_dataloader_eval, idx_test, initial_caps in zip(
        cycle_lives_all[1:], tests_dataloader_eval, idx_test_list, initial_caps_list[1:]
        ):
        capacity_trajectories_predicted_test, cap_mae_test, cap_rmse_test, cap_mape_test, \
            last_cycle_pred_test = evaluate_dataset(
                seq_to_seq_model, test_dataloader_eval, target_len, cycle_lives_test, idx_test,
                input_window=input_window, start_cycle=start_cycle, end_input_cycle=end_input_cycle,
                show_plot=False, smooth_condition=0, initial_caps = initial_caps,
                work_path=result_path + \
                    f"Figures/trail{trial_idx}/predicted_trajectories_test{test_id}.png",
            )
        cap_rmse.append(cap_rmse_test)
        cap_mae.append(cap_mae_test)
        cap_mape.append(cap_mape_test)
        life_mae_test, life_rmse_test, life_mape_test = cycle_life_metric(
            cut_off_capacity, capacity_trajectories_predicted_test,
            cycle_lives_test, idx_test, trial_idx, nominal_cap, initial_caps, last_cycle_pred_test
        )
        life_rmse.append(life_rmse_test)
        life_mae.append(life_mae_test)
        life_mape.append(life_mape_test)

        test_id += 1

    eval_time = time.time() - start_time

    prediction_output.put((f"trail #{trial_idx}", training_time, eval_time,
                           cap_rmse, cap_mae, cap_mape,
                           life_rmse, life_mae, life_mape))

def task_no_multiprocess(
    training_input_list, training_configs_list, eval_input_list, prediction_output):
    """Define a single task function defined for multiprocessing."""
    trial_idx, train_dataloader, val_dataloader, device, batch_size = training_input_list.pop()
    nominal_cap, initial_caps_list, cut_off_capacity, train_dataloader_eval, \
        tests_dataloader_eval, cycle_lives_all, idx_train, idx_test_list = eval_input_list.pop()
    layers, lr, gamma, act_func, num_epoch, early_stopping, step_start, step_patience, \
        raw_data_path, result_path, target_len, input_window, start_cycle, \
        end_input_cycle = training_configs_list.pop()
    monitor_path = result_path + f'{trial_idx}'
    if not os.path.exists(monitor_path):
        os.mkdir(monitor_path)

    # Step 1 train the model
    seq_to_seq_model = SeqToSeqLSTM(input_size=1, hidden_size=100, num_layers=4, device=device,
                                    dense_layers=layers, act_func=act_func).to(device)
    start_time = time.time()
    _, _ = seq_to_seq_model.train_encoder_decoder(
        train_dataloader, val_dataloader, n_epochs=num_epoch,
        target_len=target_len, batch_size=batch_size,
        training_prediction='mixed_teacher_forcing',
        teacher_forcing_ratio=0.6, learning_rate=lr, gamma=gamma, dynamic_tf=False,
        early_stopping=early_stopping, step_start=step_start, step_patience=step_patience,
        monitor_path=monitor_path,
    )

    training_time = time.time() - start_time

    # Step 2 Evaluate predicted capacity trajectories
    seq_to_seq_model.load_state_dict(
        torch.load(monitor_path + "/best_model.pt", map_location=torch.device('cpu'))
    )
    seq_to_seq_model.eval()

    start_time = time.time()

    # first evaluate training performance: capacity trajectory and cycle life prediction
    print("training performance")
    if not os.path.exists(result_path + f"Figures/trail{trial_idx}"):
        os.mkdir(result_path + f"Figures/trail{trial_idx}")
    capacity_trajectories_predicted_train, cap_mae_train, cap_rmse_train, cap_mape_train, \
        last_cycle_pred_train = evaluate_dataset(
            seq_to_seq_model, train_dataloader_eval, target_len, cycle_lives_all[0],
            idx_train, input_window=input_window, start_cycle=start_cycle,
            end_input_cycle=end_input_cycle, show_plot=False,
            work_path=result_path + f"Figures/trail{trial_idx}/predicted_trajectories_train.png",
            initial_caps = initial_caps_list[0],
        )
    life_mae_train, life_rmse_train, life_mape_train = cycle_life_metric(
        cut_off_capacity, capacity_trajectories_predicted_train, cycle_lives_all[0],
        idx_train, trial_idx, nominal_cap, initial_caps_list[0], last_cycle_pred_train
    )

    # create metric lists
    cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape, \
        last_cycle_pred_list = [], [], [], [], [], [], []
    last_cycle_pred_list.append(last_cycle_pred_train)
    cap_rmse.append(cap_rmse_train)
    cap_mae.append(cap_mae_train)
    cap_mape.append(cap_mape_train)
    life_rmse.append(life_rmse_train)
    life_mae.append(life_mae_train)
    life_mape.append(life_mape_train)

    # second evaluate test performance: capacity trajectory and cycle life prediction
    print("test performance")
    test_id = 1
    for cycle_lives_test, test_dataloader_eval, idx_test, initial_caps in zip(
            cycle_lives_all[1:], tests_dataloader_eval, idx_test_list, initial_caps_list[1:]
        ):
        capacity_trajectories_predicted_test, cap_mae_test, cap_rmse_test, cap_mape_test, \
            last_cycle_pred = evaluate_dataset(
                seq_to_seq_model, test_dataloader_eval, target_len, cycle_lives_test, idx_test,
                input_window=input_window, start_cycle=start_cycle, end_input_cycle=end_input_cycle,
                show_plot=False, smooth_condition=0, initial_caps = initial_caps,
                work_path=result_path + \
                    f"Figures/trail{trial_idx}/predicted_trajectories_test{test_id}.png",
            )
        last_cycle_pred_list.append(last_cycle_pred)
        cap_rmse.append(cap_rmse_test)
        cap_mae.append(cap_mae_test)
        cap_mape.append(cap_mape_test)
        life_mae_test, life_rmse_test, life_mape_test = cycle_life_metric(
            cut_off_capacity, capacity_trajectories_predicted_test,
            cycle_lives_test, idx_test, trial_idx, nominal_cap, initial_caps, last_cycle_pred
        )
        life_rmse.append(life_rmse_test)
        life_mae.append(life_mae_test)
        life_mape.append(life_mape_test)
        test_id += 1

    eval_time = time.time() - start_time
    prediction_output.append((f"trail #{trial_idx}", training_time, eval_time, cap_rmse, cap_mae,
                              cap_mape, life_rmse, life_mae, life_mape))



def evaluate_dataset(
        model, dataloader, target_len, cycle_lives, idx_test, input_window=300,
        start_cycle=0, end_input_cycle=100, smooth_condition=0, show_plot=False, work_path=None,
        initial_caps=None
    ):
    """Evaluate a dataset."""
    fig_width, fig_height = 1.8, 1.8
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    font_size = 9
    max_cycle = 2500
    colors = ['black', 'tab:blue', 'tab:orange']
    n_cols = 5
    n_rows = int(np.ceil(len(dataloader.dataset) / n_cols))
    if show_plot:
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            figsize=(fig_width * n_cols, fig_height * n_rows),
            sharex='all', sharey='all'
        )
    capacity_trajectories_predicted = []
    mae, rmse, mape= 0, 0, 0
    last_valid_pred_cycle = np.full((len(dataloader.dataset), ), np.nan)
    for num_cell, (input, target) in enumerate(dataloader):
        # get the cycle life
        cycle_life = int(cycle_lives[idx_test[num_cell]])
        # get inital cap
        intia_cap = initial_caps[idx_test[num_cell]]
        # first plot the input first 100 cycles
        cycles_input = np.arange(start_cycle, end_input_cycle + 1, 5)
        if show_plot:
            axes[num_cell // n_cols, num_cell % n_cols].plot(
                cycles_input, input[0, input_window - len(cycles_input):input_window, 0],
                linestyle=None, marker=None, markersize=2, color=colors[2],
                alpha=0.9, label='Input'
            )

        # second, plot the target output
        num_valid_values = np.where(target[0, :, 0] == 0)[0][0]
        cycles_ouput = np.arange(
            end_input_cycle + 5, end_input_cycle + 5 + num_valid_values * 50, 50
        )
        # cycles_ouput = np.arange(end_input_cycle + 5, cycle_life + 50, 50)
        if show_plot:
            axes[num_cell // n_cols, num_cell % n_cols].plot(
                cycles_ouput, target[0, 0:len(cycles_ouput), 0],
                linestyle=None, marker=None, markersize=2, color='black',
                linewidth=1,
                alpha=0.9, label='Observed'
            )

        # third, plot the predicted output
        with torch.no_grad():
            encoder_output, encoder_hidden = model.encoder(input.to(model.device).transpose(0, 1))
            decoder_input = encoder_hidden[0][-1, :, :].unsqueeze(0).expand(target_len, -1, -1)
            outputs, _ = model.decoder(decoder_input)
            outputs = outputs.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        last_valid_pred_cycle[num_cell] = end_input_cycle + 5 +\
             (np.where(outputs[:, 0, 0] <= 0.1)[0][0]) * 50

        # calculate error
        actual_caps = target[0, 0:len(cycles_ouput), 0] * intia_cap
        pred_cap = outputs[0:len(cycles_ouput), 0, 0] * intia_cap
        cap_error = actual_caps - pred_cap
        mae += (np.abs(cap_error).mean() / len(dataloader))
        mape += (np.abs(cap_error/actual_caps).mean() / len(dataloader)) * 100
        rmse += (np.sqrt(((cap_error) ** 2).mean()) / len(dataloader))

        # next, inversely interpolate the predicted capacity trajectory for cycle life cal
        cycles_ouput_pred = np.arange(end_input_cycle + 5, last_valid_pred_cycle[num_cell] + 1, 50)
        cycles_pred = np.concatenate([cycles_input, cycles_ouput_pred])
        capacities_pred = np.concatenate(
            [
                input[0, input_window - len(cycles_input):input_window, 0],
                outputs[0:len(cycles_ouput_pred), 0, 0]
            ]
        )
        capacity_trajectories_predicted.append(
            interpolate.splrep(cycles_pred, capacities_pred, s=smooth_condition)
        )
        # caps_pred = interpolate.splev(x, capacity_trajectories_predicted, der=0)
        if show_plot:
            axes[num_cell // n_cols, num_cell % n_cols].plot(
                cycles_ouput, outputs[0:len(cycles_ouput), 0, 0], marker=None,
                markersize=2, color=colors[1], linewidth=1, linestyle='--',
                alpha=0.9, label='Predicted',
            )
            axes[num_cell // n_cols, num_cell % n_cols].set_xlabel('Cycle')
            axes[num_cell // n_cols, num_cell % n_cols].set_ylabel('Normalized capacity')
            axes[num_cell // n_cols, num_cell % n_cols].grid()
            axes[num_cell // n_cols, num_cell % n_cols].set_ylim(0.8, 1.05)
            axes[num_cell // n_cols, num_cell % n_cols].set_xlim(0, max_cycle * 1.05)
            axes[num_cell // n_cols, num_cell % n_cols].set_yticks(
                np.linspace(0.8, 1, 3), fontSize=font_size
            )
            axes[num_cell // n_cols, num_cell % n_cols].set_xticks(
                [0, 1000, 2000], fontSize=font_size
            )
            axes[num_cell // n_cols, num_cell % n_cols].tick_params(
                axis='both', labelsize=font_size
            )

    if show_plot:
        line1, = axes[0, 0].plot(0, 0, color=colors[2], alpha=0.9)
        line2, = axes[0, 0].plot(0, 0, color=colors[0], alpha=0.9)
        line3, = axes[0, 0].plot(0, 0, color=colors[1], alpha=0.9, linestyle='--')
        labels = ["Input", "Observed", "Predicted"]
        fig.legend(
            [line1, line2, line3],
            labels, bbox_to_anchor=(0.5, 0.92 + 0.08 * (len(idx_test) - 10)/38),
            loc='lower center', ncol=3, frameon=True, fontsize=font_size
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.95 + 0.03 * (len(idx_test)-10)/38])
        plt.show()
        fig.savefig(work_path, transparent=True, format='png', dpi=1000, bbox_inches='tight')
    return capacity_trajectories_predicted, mae, rmse, mape, last_valid_pred_cycle
