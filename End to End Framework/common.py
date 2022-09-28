"""common fucntions and classes for the end-to-end framework."""

import time
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from train_eval import train, eval, cal_cycle_life
from end_to_end_modules import EndToEnd
from empirical_models import create_empirical_models
import numpy as np
import matplotlib.pyplot as plt


class EarlyStoppingCheck:
    """An early stropping class."""

    def __init__(self, step_start, step_patience, monitor_path, network_id):
        """Null."""
        self.step_start = step_start
        self.step_patience = step_patience
        self.early_stop = False
        self.min_val_loss = float("inf")
        self.step_min_val_loss = 0
        self.path = monitor_path + f"/best_model_NN{network_id}.pt"

    def step(self, model, val_loss, epoch):
        """Monitor one more training epoch."""
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.step_min_val_loss = epoch
            torch.save(model.state_dict(), self.path)
        if epoch < self.step_start:
            return self.early_stop, self.min_val_loss
        elif epoch - self.step_min_val_loss >= self.step_patience:
            self.early_stop = True

        return self.early_stop, self.min_val_loss


class ExpertDataSet(Dataset):
    """Build iteratable dataset."""

    def __init__(self, input_dataset, capacity_trajectories, cycle_lives, initial_capacities):
        """Null."""
        self.input = input_dataset
        self.time = capacity_trajectories[:, :, 0]
        self.capacity = capacity_trajectories[:, :, 1]
        self.diff_capacity = capacity_trajectories[:, :, 2]
        self.cycle_lives = cycle_lives
        self.initial_caps = initial_capacities

    def __getitem__(self, index):
        """Null."""
        return (
            self.input[index], self.time[index, :], self.capacity[index, :],
            self.diff_capacity[index, :], self.cycle_lives[index],
            self.initial_caps[index]
        )

    def __len__(self):
        """Null."""
        return len(self.capacity)


def get_metrics(empirical_model, empirical_models_solve, empirical_params,
                dataloader_eval_list, device, nominal_cap, cutoff_cap=0.8,
                batch_size=64, file_path=None, is_plot=False):
    """Calculate capacity-trajectory predictive metrics."""
    cap_rmse = [0] * len(dataloader_eval_list)
    cap_mae = [0] * len(dataloader_eval_list)
    cap_mape = [0] * len(dataloader_eval_list)
    cycle_lives_obs_list = []
    times_all, caps_all, caps_pred_all, initial_caps_list = [], [], [], []
    for idx, dataloader in enumerate(dataloader_eval_list):
        times, caps, caps_pred, cycle_lives, initial_caps = [], [], [], [], []
        for batch_idx, (_, time, capacity, _, cycle_life, initial_cap) in enumerate(dataloader):
            # detach to numpy
            cap_pred = empirical_model(
                time,
                empirical_params[idx][
                    batch_idx * batch_size:batch_idx * batch_size+len(time), :
                ].cpu()
            )
            cycle_lives.append(cycle_life)
            initial_caps.append(initial_cap)
            times.append(time)
            caps.append(capacity)
            caps_pred.append(cap_pred)

        cycle_lives_obs_list.append(torch.cat(cycle_lives).detach().numpy())
        initial_caps_list.append(torch.cat(initial_caps).detach().numpy())
        times_all.append(torch.cat(times).detach().numpy())
        caps_all.append(torch.cat(caps).detach().numpy())
        caps_pred_all.append(torch.cat(caps_pred).detach().numpy())
        cap_error = caps_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1) - \
                    caps_pred_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1)
        cap_rmse[idx] = np.sqrt((cap_error ** 2).mean(axis=1)).mean()
        cap_mae[idx] = np.abs(cap_error).mean()
        cap_mape[idx] = np.abs(
            cap_error/(
                caps_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1)
            )
        ).mean() * 100

    # cycle life predictive metrics
    life_rmse = [0] * len(dataloader_eval_list)
    life_mae = [0] * len(dataloader_eval_list)
    life_mape = [0] * len(dataloader_eval_list)
    for idx, dataloader in enumerate(dataloader_eval_list):
        cycle_lives_pred = cal_cycle_life(
            cycle_lives_obs_list[idx], empirical_params[idx], empirical_models_solve,
            nominal_cap, initial_caps_list[idx], life_threshold=cutoff_cap
        )
        life_error = cycle_lives_pred - cycle_lives_obs_list[idx]
        life_rmse[idx] = np.sqrt((life_error ** 2).mean())
        life_mae[idx] = np.abs(life_error).mean()
        life_mape[idx] = np.abs(life_error/cycle_lives_obs_list[idx]).mean() * 100
    if is_plot:
        for figure_num, _ in enumerate(times_all):
            times = times_all[figure_num]
            caps, caps_pred = caps_all[figure_num], caps_pred_all[figure_num]
            fig_width, fig_height = 2.5, 2.5
            colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:red']
            max_time = max([np.max(time_) for time_ in times])
            ncols = 5
            nrows = int(np.ceil(len(times) / ncols))
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(fig_width * ncols, fig_height * nrows),
            )
            # This is the plot for daily revenues
            for num_plot in range(len(times)):
                axes[num_plot // ncols, num_plot % ncols].plot(
                    times[num_plot, :], caps[num_plot, :],
                    linestyle=None, marker='o', markersize=2, color=colors[0],
                    alpha=0.7, label='Observed'
                )
                axes[num_plot // ncols, num_plot % ncols].plot(
                    times[num_plot, :], caps_pred[num_plot, :], marker='>',
                    markersize=2, color=colors[1], alpha=0.7, label='Predicted'
                )
                axes[num_plot // ncols, num_plot % ncols].set_xlabel('Cycle')
                axes[num_plot // ncols, num_plot % ncols].set_ylabel('Normalized capacity')
                axes[num_plot // ncols, num_plot % ncols].grid()
                axes[num_plot // ncols, num_plot % ncols].set_yticks(np.linspace(0.6, 1, 5))
                axes[num_plot // ncols, num_plot % ncols].set_ylim(0.6, 1.05)
                axes[num_plot // ncols, num_plot % ncols].set_xlim(0, max_time)

            axes[0, 0].legend()
            plt.suptitle("Test dataset (30% of the total cells)", y=1.03)
            fig.tight_layout(rect=[0, 0.01, 1, 0.99])
            plt.show()
            fig.savefig(
                file_path + '_dataset_' + str(figure_num) + '.png',
                transparent=True, format='png', dpi=1000, bbox_inches='tight'
            )

    return cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape


def device_info():
    """Get device info."""
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    return device, kwargs


def task(training_input_queue, training_configs_queue, eval_input_queue, prediction_output):
    """
    Dfeine a function for multiprocessing run.

    This function define a single task of training ensemble neural network and
    evaluate the ensemble network's predictive performance on capacity trajectory
    and cycle life.
    """
    trial_idx, train_dataloader, val_dataloader, device = training_input_queue.get()
    model_name, limits_coeff, input_size, number_nn, layers, lr, gamma, act_func, ratios,\
    loss_func, num_epoch, early_stopping, step_start, step_patience, raw_data_path, \
    result_path = training_configs_queue.get()
    nominal_cap, cut_off_cap, dataloader_eval_list = eval_input_queue.get()
    params, empirical_model, empirical_models_solve, \
        diff_empirical_model, params_lbs_ubs = create_empirical_models(
        raw_data_path, model_name, limits_coeff, device
        )

    # Step 1 create models
    models = []
    start_time = time.time()
    for network_id in range(number_nn):
        model = EndToEnd(
            input_size, layers, params, empirical_model,
            diff_empirical_model, params_lbs_ubs, act_func,
        ).to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
        monitor_path = result_path + f'trail {trial_idx}'
        # monitor_path = f'C:/tmp_results/trail {trial_idx}'
        models.append(model)

        # Step 2 train the model
        if early_stopping:
            early_stop = EarlyStoppingCheck(step_start, step_patience, monitor_path, network_id)
            stopping_status = False
            if not os.path.exists(monitor_path):
                os.makedirs(monitor_path)
        for epoch in range(1, num_epoch+1):
            train(model, train_dataloader, loss_func, optimizer, ratios, device)
            scheduler.step()
            test_loss1, test_loss2, test_loss3, test_loss4, _ = eval(
                model, val_dataloader, loss_func, ratios, device)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}\n-------------------------------")
                print(f"Validation set: mean Q Error, Q_end error, mean dQ/dt error, dQ/dt at end error, : "
                      f"{test_loss1:.4f}, {test_loss2:.4f}, {test_loss3:.4f}, {test_loss4:.4f}")
            val_metric = ratios[0] * test_loss1 + ratios[1] * test_loss2 +\
                         ratios[2] * test_loss3 + ratios[3] * test_loss4
            stopping_status, _ = early_stop.step(model, val_metric, epoch)
            if stopping_status:
                print(f"Early stopping! The validation error has not improved for the past {step_patience} epochs")
                break
        if not stopping_status:
            print(f"Done! All {num_epoch} training epochs completed!")
        torch.save(model.state_dict(), monitor_path + f"/best_model_NN{network_id}.pt")
    training_time = time.time() - start_time

    # Step 3 Evaluate predicted empirical models
    start_time = time.time()
    empirical_params = []
    for network_id in range(number_nn):
        print(f"----------------{network_id}-------------------")
        models[network_id].load_state_dict(
            torch.load(monitor_path + f"/best_model_NN{network_id}.pt")
        )
        models[network_id].eval()
        for idx, dataloader in enumerate(dataloader_eval_list):
            _, _, _, _, empirical_params_ = eval(
                models[network_id], dataloader, loss_func, ratios, device
            )
            if network_id == 0:
                empirical_params.append(empirical_params_ / number_nn)
            else:
                empirical_params[idx] += empirical_params_ / number_nn
    # save empirical params
    for idx_data, empirical_param in enumerate(empirical_params):
        if not os.path.exists(result_path + 'Empirical params/run' + str(trial_idx)):
            os.makedirs(result_path + 'Empirical params/run' + str(trial_idx))
        np.savetxt(
            result_path + 'Empirical params/run' + str(trial_idx) + \
                '/dataset_' + str(idx_data) + '.csv',
            empirical_param.numpy(), delimiter=','
        )
    eval_time = time.time() - start_time
    # summarize metric
    file_path = result_path + 'Figures/run' + str(trial_idx)
    cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape = get_metrics(
        empirical_model, empirical_models_solve, empirical_params,
        dataloader_eval_list, device, nominal_cap, cutoff_cap=cut_off_cap,
        file_path=file_path, is_plot=False)
    prediction_output.put(
        (
            f"trail #{trial_idx}", training_time, eval_time,
            cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape,
        )
    )

def task_no_multiprocess(
        training_input_list, training_configs_list, eval_input_list, prediction_output
    ):
    """
    Define a similar task function for singleprocessing run.

    This function define a single task of training ensemble neural network and
    evaluate the ensemble network's predictive performance on capacity trajectory
    and cycle life.
    """
    trial_idx, train_dataloader, val_dataloader, device = training_input_list.pop()
    model_name, limits_coeff, input_size, number_nn, layers, lr, gamma, act_func, ratios,\
    loss_func, num_epoch, early_stopping, step_start, step_patience, raw_data_path, \
    result_path = training_configs_list.pop()
    nominal_cap, cut_off_cap, dataloader_eval_list = eval_input_list.pop()
    params, empirical_model, empirical_models_solve, diff_empirical_model, \
        params_lbs_ubs = create_empirical_models(
            raw_data_path, model_name, limits_coeff, device
        )

    # Step 1 create models
    models = []
    start_time = time.time()
    for network_id in range(number_nn):
        model = EndToEnd(
            input_size, layers, params, empirical_model, diff_empirical_model,
            params_lbs_ubs, act_func,
        ).to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
        monitor_path = f'C:/tmp_results/trail {trial_idx}'
        models.append(model)

        # Step 2 train the model
        if early_stopping:
            early_stop = EarlyStoppingCheck(step_start, step_patience, monitor_path, network_id)
            stopping_status = False
            if not os.path.exists(monitor_path):
                os.makedirs(monitor_path)
        for epoch in range(1, num_epoch+1):
            # print(f"Epoch {epoch}\n-------------------------------")
            train(model, train_dataloader, loss_func, optimizer, ratios, device)
            scheduler.step()
            test_loss1, test_loss2, test_loss3, test_loss4, _ = eval(
                model, val_dataloader, loss_func, ratios, device)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}\n-------------------------------")
                print(f"Validation set: mean Q Error, Q_end error, mean dQ/dt error, dQ/dt at end error, : "
                      f"{test_loss1:.4f}, {test_loss2:.4f}, {test_loss3:.4f}, {test_loss4:.4f}")
            val_metric = ratios[0] * test_loss1 + ratios[1] * test_loss2 +\
                         ratios[2] * test_loss3 + ratios[3] * test_loss4
            stopping_status, _ = early_stop.step(model, val_metric, epoch)
            if stopping_status:
                print(f"Early stopping! The validation error has not improved for the past {step_patience} epochs")
                break
        if not stopping_status:
            print(f"Done! All {num_epoch} training epochs completed!")
        torch.save(model.state_dict(), monitor_path + f"/best_model_NN{network_id}.pt")

    training_time = time.time() - start_time

    # Step 3 Evaluate predicted empirical models
    start_time = time.time()
    empirical_params = []
    for network_id in range(number_nn):
        print(f"-----------------------{network_id}-----------------------------")
        models[network_id].load_state_dict(
            torch.load(monitor_path + f"/best_model_NN{network_id}.pt")
        )
        models[network_id].eval()
        for idx, dataloader in enumerate(dataloader_eval_list):
            _, _, _, _, empirical_params_ = eval(
                models[network_id], dataloader, loss_func, ratios, device
            )
            if network_id == 0:
                empirical_params.append(empirical_params_ / number_nn)
            else:
                empirical_params[idx] += empirical_params_ / number_nn
    eval_time = time.time() - start_time
    # summarize metric
    file_path = result_path + 'Figures/run' + str(trial_idx)
    cap_rmse, cap_mae, cap_mape, life_rmse, life_mae, life_mape = get_metrics(
        empirical_model, empirical_models_solve,empirical_params,
        dataloader_eval_list, device, nominal_cap, cutoff_cap=cut_off_cap,
        file_path=file_path, is_plot=False
    )
    prediction_output.append((f"trail #{trial_idx}", training_time, eval_time, cap_rmse, cap_mae,
                              cap_mape, life_rmse, life_mae, life_mape))
