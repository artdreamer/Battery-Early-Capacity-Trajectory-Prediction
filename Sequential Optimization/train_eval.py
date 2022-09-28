"""Train and eval modules for the sequential optimization frammework."""
import torch
import numpy as np
from scipy.optimize import fsolve


def train(model, train_loader, loss_func, optimizer, device, empirical_model_name):
    """Train the model."""
    torch.autograd.set_detect_anomaly(True)
    log_interval = 10
    size = len(train_loader.dataset)
    model.train()
    for batch_idx, (input, fitted_params, _, _, _, _, _) in enumerate(train_loader):
        # Attach the data to the device
        input, fitted_params = input.to(device), fitted_params.to(device)
        batch_size = len(input)
        # Compute prediction errors
        params_pred = model(input)
        # for power-law model
        if empirical_model_name == "power_law1":
            fitted_params[:, 0] = torch.log10(fitted_params[:, 0])
        elif empirical_model_name == "exp_linear2":
            # for exp-linear-2 model
            fitted_params = torch.log10(fitted_params)
        loss = loss_func(params_pred, fitted_params)  # error on the predicted params
        # loss.requires_grad = True
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(params)

        # print log
        if batch_idx % log_interval == 0:
            print(
                "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * batch_size,
                    size,
                    100.0 * batch_idx * batch_size / size,
                    loss.item(),
                )
            )



def eval(model, test_loader, loss_func, device, empirical_model_name):
    """Eval the trained model."""
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss = 0
    # cycle_lives_obs = []
    # capacities = []
    # capacities_pred = []
    params_list = []
    with torch.no_grad():
        for _, (input, fitted_params, _, _, _, _, _) in enumerate(test_loader):
            # Attach the data to the device
            input, fitted_params = input.to(device), fitted_params.to(device)
            # for power law model
            if empirical_model_name == "power_law1":
                fitted_params[:, 0] = torch.log10(fitted_params[:, 0])
            elif empirical_model_name == "exp_linear2":
                # for exp-linear-2 model
                fitted_params = torch.log10(fitted_params)
            # Compute prediction error
            params_pred = model(input)
            # test_loss += loss_fn(capacity_pred, capacity)
            test_loss = test_loss + loss_func(fitted_params, params_pred)

            params_list.append(params_pred)
    test_loss /= num_batches
    return test_loss, torch.cat(params_list)


def helper(cycle, empirical_models_solve, empirical_params, life_threshold):
    """Wrap the empirical capacity fade model to be used in fsolve."""
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
            helper, 3000,
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
    """Calculate cycle life predictive metrics."""
    cycle_lives_pred = cal_cycle_life(
        cycle_lives_obs, empirical_params, empirical_models_solve, nominal_cap,
        initial_caps, life_threshold
    )
    if metric_name == 'MAE':
        return np.abs(cycle_lives_pred - cycle_lives_obs.detach().cpu().numpy()).mean()
    if metric_name == 'RMSE':
        return np.square(cycle_lives_pred - cycle_lives_obs.detach().cpu().numpy()).mean() ** 0.5
