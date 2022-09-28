"""Generate prediction results including plots."""
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from common import device_info
import torch
from scipy.optimize import fsolve, root_scalar
from data_preprocessing import prepare_data_train
import matplotlib.pyplot as plt
from empirical_models import exponential_linear_capacity_fade_model2, \
    power_law_capacity_fade_model, power_law_capacity_fade_model_solve, \
    exponential_linear_capacity_fade_model2_solve
from train_eval import helper


def cal_cycle_life(
    cycle_lives_obs, empirical_params, empirical_models_solve,
    nominal_cap, inital_capcities, life_threshold=0.7
    ):
    """Calculate predictive cycle life."""
    if not isinstance(cycle_lives_obs, np.ndarray):
        cycle_lives_obs = cycle_lives_obs.detach().cpu().numpy()
    if not isinstance(empirical_params, np.ndarray):
        empirical_params = empirical_params.detach().cpu().numpy()
    # empirical_model = empirical_models_solve
    cycle_lives_pred = np.zeros((cycle_lives_obs.shape[0],))
    for num_cell, empirical_params_cell in enumerate(empirical_params):
        cycle_lives_pred[num_cell] = fsolve(
            helper, cycle_lives_obs[num_cell], args=(empirical_models_solve, empirical_params_cell,
            life_threshold * nominal_cap / inital_capcities[num_cell])
        )
        # start, end = 100, 5000
        # if helper(start, empirical_models_solve, empirical_params_cell,
        #           life_threshold * nominal_cap / inital_capcities[num_cell]) * \
        #     helper(end, empirical_models_solve, empirical_params_cell,
        #             life_threshold * nominal_cap / inital_capcities[num_cell]) >= 0:
        #     print(num_cell, empirical_params_cell)
        # 
        # cycle_lives_pred[num_cell] = root_scalar(
        #     helper, 
        #     args=(
        #         empirical_models_solve, empirical_params_cell, 
        #         life_threshold * nominal_cap / inital_capcities[num_cell]
        #     ),
        #     bracket=(start, end)
        # ).root
        # 
        # cycle_lives_pred[num_cell] = 10 ** (
        #     (
        #         np.log10(1 - life_threshold * nominal_cap / inital_capcities[num_cell]) - \
        #             np.log10(empirical_params_cell[0])
        #     ) / empirical_params_cell[1]
        # )
    return  cycle_lives_pred

def get_plots(
        empirical_model, empirical_models_solve, empirical_params, dataloader_eval_list,
        nominal_cap, cutoff_cap=0.8, batch_size=64, file_path=None, life_result_path=None,
        metric_result_path=None, caps_result_path=None, is_plot=False
    ):
    """Generate plots for an evaluation dataset."""
    # caps and caps_pred
    cycle_lives_obs_list = []
    cap_rmse = [0] * len(dataloader_eval_list)
    cap_mae = [0] * len(dataloader_eval_list)
    cap_mape = [0] * len(dataloader_eval_list)
    times_all, caps_all, caps_pred_all, initial_caps_list = [], [], [], []
    caps_abs_all, caps_pred_abs_all = [], []
    for idx, dataloader in enumerate(dataloader_eval_list):
        times, caps, caps_pred, cycle_lives, initial_caps = [], [], [], [], []
        caps_pred_abs, caps_abs = [], []
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
        caps_abs =  caps_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1)
        caps_pred_abs = caps_pred_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1)
        caps_abs_all.append(caps_abs)
        caps_pred_abs_all.append(caps_pred_abs)
        cap_error = caps_abs - caps_pred_abs
        cap_rmse[idx] = np.sqrt((cap_error ** 2).mean(axis=1))
        cap_mae[idx] = np.abs(cap_error).mean(axis=1)
        cap_mape[idx] = np.abs(
            cap_error / (caps_all[-1] * np.expand_dims(initial_caps_list[-1], axis=1))
        ) * 100
    with open(caps_result_path+"/times_caps_caps_pred.pkl", "wb") as file:
        pickle.dump([times_all, caps_abs_all, caps_pred_abs_all], file)

    with open(metric_result_path+"/Q_rmse_mae_mape.pkl", "wb") as file:
        pickle.dump([cap_rmse, cap_mae, cap_mape], file)

    # cycle life predictive metrics
    life_ae = [0] * len(dataloader_eval_list)
    cycle_lives_pred_all = []
    for idx, dataloader in enumerate(dataloader_eval_list):
        cycle_lives_pred = cal_cycle_life(
            cycle_lives_obs_list[idx], empirical_params[idx], empirical_models_solve,
            nominal_cap, initial_caps_list[idx], life_threshold=0.8
        )
        cycle_lives_pred_all.append(cycle_lives_pred)
        cycle_life_errors = cycle_lives_obs_list[idx] - cycle_lives_pred
        if life_result_path:
            fname = life_result_path + str(idx) + ".csv"
            np.savetxt(fname, np.stack((cycle_lives_obs_list[idx], cycle_lives_pred), axis=1))

        if metric_result_path:
            life_ae[idx] = np.abs(cycle_life_errors)

    with open(metric_result_path+"/life_ae.pkl", "wb") as file:
        pickle.dump(life_ae, file)


    if is_plot:
        font_size = 9
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
        # plt.rcParams["legend.loc"] = 'lower left'
        for figure_num, _ in enumerate(times_all):
            times, caps = times_all[figure_num], caps_all[figure_num]
            cycle_lives_pred = cycle_lives_pred_all[figure_num]
            cycle_lives = cycle_lives_obs_list[figure_num]
            fig_width, fig_height = 1.8, 1.8
            colors = ['black', 'tab:blue', 'tab:orange', 'tab:red']
            max_time_obs_all = max([np.max(time_) for time_ in times])
            max_time_pred_all = cycle_lives_pred.max()
            max_time = max([max_time_obs_all, max_time_pred_all])
            max_time = max_time_obs_all
            ncols = 5
            nrows = int(np.ceil(len(times) / ncols))
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(fig_width * ncols, fig_height * nrows)
            )
            # This is the plot for daily revenues
            for num_plot in range(len(times)):
                axes[num_plot // ncols, num_plot % ncols].plot(
                    times[num_plot, :], caps[num_plot, :], linestyle=None, linewidth=1,
                    marker=None, markersize=2, color=colors[0], alpha=0.9, label='Observed'
                )
                max_time_pred = cycle_lives_pred[num_plot]
                time_pred = torch.arange(0, max(max_time_pred, max_time) * 1.05, 10)
                cap_pred = empirical_model(
                    time_pred.reshape(1, -1),
                    empirical_params[figure_num][num_plot:num_plot+1, :].cpu()
                )
                cap_pred = cap_pred.squeeze(dim=0)
                axes[num_plot // ncols, num_plot % ncols].plot(
                    time_pred, cap_pred, linestyle='--', linewidth=2, marker=None,
                    markersize=1, color=colors[1], alpha=0.9, label='Predicted'
                )
                axes[num_plot // ncols, num_plot % ncols].set_xlabel('Cycle')
                axes[num_plot // ncols, num_plot % ncols].set_ylabel('Normalized capacity')
                axes[num_plot // ncols, num_plot % ncols].grid()
                axes[num_plot // ncols, num_plot % ncols].set_yticks(
                    np.linspace(0.6, 1, 3), fontSize=font_size
                )
                axes[num_plot // ncols, num_plot % ncols].set_xticks(
                    [0, 1000, 2000], fontSize=font_size
                )
                axes[num_plot // ncols, num_plot % ncols].tick_params(
                    axis='both', labelsize=font_size
                )
                axes[num_plot // ncols, num_plot % ncols].set_ylim(0.6, 1.05)
                axes[num_plot // ncols, num_plot % ncols].set_xlim(0, max_time * 1.05)
            # axes[0, 0].legend()
            line1, = axes[0, 0].plot(0, 0, color=colors[0])
            line2, = axes[0, 0].plot(0, 0, color=colors[1], linestyle='--')
            labels = ["Observed", "Predicted"]
            fig.legend(
                [line1, line2], labels, bbox_to_anchor=(0.5, 0.975), loc = 'lower center',
                ncol=2, frameon=True, fontsize=font_size
            )
            # plt.suptitle("Test dataset (30% of the total cells)", y=1.03)
            fig.tight_layout(rect=[0, 0.01, 1, 0.98])
            plt.show()
            fig.savefig(
                file_path + 'dataset_' + str(figure_num) + '.png', transparent=False,
                format='png', dpi=1000, bbox_inches='tight'
            )


def main_plot(dataset, model_name):
    """Generate plots for a model on a dataset."""
    empirical_model = power_law_capacity_fade_model if dataset == "169 LFP" \
        else exponential_linear_capacity_fade_model2
    empirical_models_solve = power_law_capacity_fade_model_solve if dataset == "169 LFP" \
        else exponential_linear_capacity_fade_model2_solve

    dataset_idxs = range(4) if dataset == "169 LFP" else range(2)
    nominal_cap = 1.1 if dataset == "169 LFP" else 1.85
    model_name_str_list = model_name.split("-")
    if model_name_str_list[-1] == "deltaQ":
        input_type = "delta QV 100-10"
    elif model_name_str_list[-1] == "discharge":
        input_type = "discharge features"

    raw_data_path = '../dataset/' + dataset
    data_path = "All empirical params/"
    workpath = "All plots/"
    life_result_path = "All cycle lives/" + dataset + '/' + model_name
    metric_result_path = "All metrics/" + dataset + '/' + model_name
    caps_result_path = "All predictions/" + dataset + '/' + model_name
    if not os.path.exists(life_result_path):
        os.mkdir(life_result_path)
    life_result_path = life_result_path + "/dataset_"
    # Step 1: load empirical model params
    file_path = data_path + dataset + '/' + model_name + '/dataset_'
    params = []
    for idx in dataset_idxs:
        param = np.genfromtxt(file_path + str(idx) + '.csv', delimiter=',', dtype="float32")
        param_tensor = torch.from_numpy(param)
        # param_tensor[:, 0] = 10 ** param_tensor[:, 0]
        # param_tensor = param_tensor
        params.append(param_tensor)

    # Step 2: generate eval datasets
    precision = "float32"
    torch.set_default_dtype(torch.float32)
    _, _, _, train_dataset, test_datast_list = prepare_data_train(
        dataset, input_type, raw_data_path, precision,batch_size=64
    )

    # create eval dataloaders, not shuffle!!!
    _, kwargs = device_info()
    batch_size = 64
    dataloader_eval_list = [
        DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )
    ]
    for test_dataset in test_datast_list:
        dataloader_eval_list.append(
            DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs
            )
        )

    # Step 3: plot
    file_path = workpath + dataset + '/' + model_name + '/'
    get_plots(
        empirical_model, empirical_models_solve, params, dataloader_eval_list,
        nominal_cap, cutoff_cap=0.8, batch_size=64, file_path=file_path,
        life_result_path=life_result_path, metric_result_path=metric_result_path,
        caps_result_path=caps_result_path, is_plot=False
    )

if __name__ == "__main__":
    datasets = ["48 NMC"]
    model_names = [
        # "E2E-Elastic-discharge",
        # "E2E-Elastic-deltaQ",
        "E2E-ENN-discharge",
        # "E2E-ENN-deltaQ"
        # "Sequential-Elastic-discharge",
        # "Sequential-Elastic-deltaQ",
        # "Sequential-ENN-discharge",
        # "Sequential-ENN-deltaQ",
    ]

    # inputs
    for dataset in datasets:
        for model_name in model_names:
            main_plot(dataset, model_name)
