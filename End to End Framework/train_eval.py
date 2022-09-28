"""Train and eval functions."""
import torch
import numpy as np
from scipy.optimize import fsolve


def train(model, train_loader, loss_func, optimizer, ratios, device):
    """Train the model."""
    log_interval = 10
    size = len(train_loader.dataset)
    model.train()
    for batch_idx, (input, time, capacity, diff_capacity, _, _) in enumerate(train_loader):
        # Attach the data to the device
        input, time, capacity = input.to(device), time.to(device), capacity.to(device)
        diff_capacity = diff_capacity.to(device)
        batch_size = len(capacity)
        # Compute prediction errors
        capacity_pred, diff_capacity_pred, _ = model(input, time)
        # error on the entire capacity trajectory
        loss1 = loss_func(capacity_pred, capacity)
        # error on the end point of the trajectory
        loss2 = loss_func(capacity_pred[:, -1], capacity[:, -1])
        # error on the entire diff_capacity
        loss3 = loss_func(diff_capacity_pred, diff_capacity)
        # error on the end point of the diff_capacity
        loss4 = loss_func(diff_capacity_pred[:, -1], diff_capacity[:, -1])
        loss = ratios[0] * loss1 + ratios[1] * loss2 + ratios[2] * loss3 + ratios[3] * loss4
        # loss.requires_grad = True

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(params)

        # print log
        # if batch_idx % log_interval == 0:
        #     print(
        #         "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             batch_idx * batch_size,
        #             size,
        #             100.0 * batch_idx * batch_size / size,
        #             loss.item(),
        #         )
        #     )



def eval(model, test_loader, loss_func, ratios, device):
    """Eval the trained model."""
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss1, test_loss2, test_loss3, test_loss4 = 0, 0, 0, 0
    params_list = []
    with torch.no_grad():
        for _, (input, time, capacity, diff_capacity, _, _) in enumerate(test_loader):
            # Attach the data to the device
            input, time, capacity, diff_capacity = input.to(device), time.to(device), \
                                                   capacity.to(device), diff_capacity.to(device)
            # Compute prediction error
            capacity_pred, diff_capacity_pred, params_cell = model(input, time)
            test_loss1 = test_loss1 + loss_func(capacity_pred, capacity)
            test_loss2 = test_loss2 + loss_func(capacity_pred[:, -1], capacity[:, -1])
            test_loss3 = test_loss3 + loss_func(diff_capacity_pred, diff_capacity)
            test_loss4 = test_loss4 + loss_func(diff_capacity_pred[:, -1], diff_capacity[:, -1])
            params_list.append(params_cell)
    test_loss1 /= num_batches
    test_loss2 /= num_batches
    test_loss3 /= num_batches
    test_loss4 /= num_batches
    test_loss = ratios[0] * test_loss1 + ratios[1] * test_loss2 + \
        ratios[2] * test_loss3 + ratios[3] * test_loss4
    return test_loss1, test_loss2, test_loss3, test_loss4, torch.cat(params_list)


def helper(cycle, empirical_models_solve, empirical_params, life_threshold):
    """Wrap the empirical capacity fade model to be used in calculating the cycle life."""
    return empirical_models_solve(cycle, empirical_params) - life_threshold


def cal_cycle_life(
    cycle_lives_obs, empirical_params, empirical_models_solve, nominal_cap,
    inital_capcities, life_threshold=0.7
    ):
    """Calculate predictive cycle life."""
    if not isinstance(cycle_lives_obs, np.ndarray):
        cycle_lives_obs = cycle_lives_obs.detach().cpu().numpy()
    if not isinstance(empirical_params, np.ndarray):
        empirical_params = empirical_params.detach().cpu().numpy()
    cycle_lives_pred = np.zeros((cycle_lives_obs.shape[0],))
    for num_cell, empirical_params_cell in enumerate(empirical_params):
        cycle_lives_pred[num_cell] = fsolve(
            helper, cycle_lives_obs[num_cell],
            args=(
                empirical_models_solve, empirical_params_cell,
                life_threshold * nominal_cap / inital_capcities[num_cell]
            )
        )
    return  cycle_lives_pred


def life_metric(
        cycle_lives_obs, empirical_params, empirical_models_solve, nominal_cap,
        initial_caps, life_threshold=0.7, metric_name='MAE'
    ):
    """
    Calculate cycle life predictive metrics.

    :param cycle_lives_obs: observed cycle lives
    :param empirical_params: predicted empirical model params
    :param empirical_models_solve: empirical models using np array format
    :param nominal_cap: nominal capacity
    :param life_threshold: Normalized capacity threshold at EoL
    :param metric_name:
    :return:
    """
    cycle_lives_pred = cal_cycle_life(
        cycle_lives_obs, empirical_params, empirical_models_solve, nominal_cap,
        initial_caps, life_threshold
    )
    if metric_name == 'MAE':
        return np.abs(cycle_lives_pred - cycle_lives_obs.detach().cpu().numpy()).mean()
    if metric_name == 'RMSE':
        return np.square(cycle_lives_pred - cycle_lives_obs.detach().cpu().numpy()).mean() ** 0.5
