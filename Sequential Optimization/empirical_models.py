"""
Define empirical capacity fade models used for the end-to-end framework.

Models are defined in different formats.
"""
import numpy as np
import torch


def linear_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using a linear model.

    :param k: cycle number
    :param params: linear model params
    :return: discharge capacity after k cycles
    """
    a = params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return a - torch.matmul(b, ones_suplimentary) * k


def exponential_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using an exp model with two exp components.

    :param k: cycle number
    :params: a, b, c, d, parameters of the exponential model
    :return: discharge capacity after k cycles
    """
    a = params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    c = params[:, 2].reshape(-1, 1)
    d = params[:, 3].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return torch.matmul(a, ones_suplimentary) * torch.exp(torch.matmul(-b, ones_suplimentary) * k) \
           - torch.matmul(c, ones_suplimentary) * torch.exp(torch.matmul(d, ones_suplimentary) * k)


def exponential_linear_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using a exponential/linear-1 hybrid model.

    :param k: cycle number
    :param params: a, b, c, parameters of the hybrid model
    :return: discharge capacity after k cycles
    """
    a = params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    c = params[:, 2].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return 1 - torch.matmul(a, ones_suplimentary) * \
        (1 - torch.exp(- torch.matmul(b, ones_suplimentary) * k)) - \
        torch.matmul(c, ones_suplimentary) * k


def power_law_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using a power-law model.

    :param k: cycle number
    :param params: a, b: parameters of the power-law model
    :return: discharge capacity after k cycles
    """
    a = 10 ** params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return 1 - torch.matmul(a, ones_suplimentary) * torch.pow(k, torch.matmul(b, ones_suplimentary))


def exponential_power_law_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using a exponential/power-law hybrid model.

    :param k: cycle number
    :param params: a, b, c, d, parameters of the hybrid model
    :return: discharge capacity after k cycles
    """
    a = params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    c = params[:, 2].reshape(-1, 1)
    d = params[:, 3].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return torch.matmul(a, ones_suplimentary) * \
        torch.exp(torch.matmul(-b, ones_suplimentary) * k) - \
        torch.matmul(c, ones_suplimentary) * \
        torch.pow(k, torch.matmul(d, ones_suplimentary))


def power_law_two_capacity_fade_model(k, params):
    """
    Calculate the capacity fade by using a power-law model considering the knee point.

    :param k: cycle number
    :param params: a, b, parameters of the power-law model with two power terms
    :return: discharge capacity after k cycles
    """
    a = params[:, 0].reshape(-1, 1)
    b = params[:, 1].reshape(-1, 1)
    c = params[:, 2].reshape(-1, 1)
    d = params[:, 3].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return 1 - torch.matmul(a, ones_suplimentary) * \
        torch.pow(k, torch.matmul(b, ones_suplimentary)) - \
        torch.matmul(c, ones_suplimentary) * \
        torch.pow(k, torch.matmul(d, ones_suplimentary))


def exponential_linear_capacity_fade_model2(k, params):
    """
    Calculate the capacity fade by using a exponential/linear hybrid-2 model.

    :param k: cycle number
    :param params: a, b, c, parameters of the hybrid model
    :return: discharge capacity after k cycles
    """
    a = 10 ** params[:, 0].reshape(-1, 1)
    b = 10 ** params[:, 1].reshape(-1, 1)
    c = 10 ** params[:, 2].reshape(-1, 1)
    ones_suplimentary = torch.ones((1, k.shape[1])).to(params.device)
    return 1 - torch.matmul(a, ones_suplimentary) * \
        torch.exp(torch.matmul(b, ones_suplimentary) * k) \
        - torch.matmul(c, ones_suplimentary) * k


# empirical capacity fade models used cycle life prediction in np.array format
def linear_capacity_fade_model_solve(k, params):
    """
    Linear capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = params[0]
    b = params[1]
    return a - b * k


def exponential_capacity_fade_model_solve(k, params):
    """
    Exponential capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return a * np.exp(-b * k) -c * np.exp(d * k)


def exponential_linear_capacity_fade_model_solve(k, params):
    """
    exponential/linear capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = params[0]
    b = params[1]
    c = params[2]
    return 1 - a * (1 - np.exp(-b * k)) - c * k


def power_law_capacity_fade_model_solve(k, params):
    """
    power_law capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = 10 ** params[0]
    b = params[1]
    return 1 - a * np.power(k, b)


def exponential_power_law_capacity_fade_model_solve(k, params):
    """
    exponential_power_law capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return a * np.exp(-b * k) - c * np.power(k, d)


def power_law_two_capacity_fade_model_solve(k, params):
    """
    power_law_2 capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return 1 - a * np.power(k, b) - c * np.power(k, d)


def exponential_linear_capacity_fade_model2_solve(k, params):
    """
    Exponential_linear_2 capacity fade model for cycle life calculation.

    :param k:
    :param params:
    :return:
    """
    a = 10 ** params[0]
    b = 10 ** params[1]
    c = 10 ** params[2]
    return 1 - a * np.exp(b * k) - c * k


def params_limits(work_path, limits_coefficient, empirical_models, params_lbs_ubs):
    """
    Load fitted empirical model parameters.

    :param work_path:
    :param limits_coefficient:
    :param empirical_models:
    :param params_lbs_ubs:
    :return:
    """
    fitted_params = {}
    for empirical_model in empirical_models:
        fitted_params[empirical_model] = np.genfromtxt(
            work_path + '/fitted empirical models/' + empirical_model + ' parameters.csv',
            delimiter=',', dtype="float32"
        )
        params_lbs_ubs[empirical_model] = [
            torch.from_numpy(
                np.nanmin(fitted_params[empirical_model], axis=0) - \
                np.abs(np.nanmin(fitted_params[empirical_model], axis=0)) * limits_coefficient
            ),
            torch.from_numpy(
                np.nanmax(fitted_params[empirical_model], axis=0) +
                np.abs(np.nanmax(fitted_params[empirical_model], axis=0)) * limits_coefficient
            )
        ]
    return params_lbs_ubs



def create_empirical_models(model_name, work_path, limits_coefficient, device):
    """Creat empirical model.:return."""
    empirical_models = {
        "linear": linear_capacity_fade_model,
        "exp": exponential_capacity_fade_model,
        "exp_linear": exponential_linear_capacity_fade_model,
        "power_law1": power_law_capacity_fade_model,
        "power_law2": power_law_two_capacity_fade_model,
        "exp_power_law": exponential_power_law_capacity_fade_model,
        "exp_linear2": exponential_linear_capacity_fade_model2
    }
    empirical_models_solve = {
        "linear": linear_capacity_fade_model_solve,
        "exp": exponential_capacity_fade_model_solve,
        "exp_linear": exponential_linear_capacity_fade_model_solve,
        "power_law1": power_law_capacity_fade_model_solve,
        "power_law2": power_law_two_capacity_fade_model_solve,
        "exp_power_law": exponential_power_law_capacity_fade_model_solve,
        "exp_linear2": exponential_linear_capacity_fade_model2_solve
    }
    params = {
        "linear": ['a', 'b'],
        "exp": ['a', 'b', 'c', 'd'],
        "exp_linear": ['a', 'b', 'c'],
        "power_law1": ['a', 'b'],
        "power_law2": ['a', 'b', 'c', 'd'],
        "exp_power_law": ['a', 'b', 'c', 'd'],
        "exp_linear2": ['a', 'b', 'c']
    }

    params_lbs_ubs = {
        "linear": [torch.tensor([0., 0.]), torch.tensor([1e3, 1e3])],
        "exp": [torch.tensor([0., 0., 0., 0.]), torch.tensor([1e3, 1e1, 1e3, 1])],
        "exp_linear": [torch.tensor([0., 0., 0.]), torch.tensor([1e3, 10, 1e3])],
        "power_law1": [torch.tensor([-25.3395, 1.1298]), torch.tensor([-3.4976, 7.0865])],
        "power_law2": [torch.tensor([0., 0., 0., 1.]), torch.tensor([1e6, 1, 1e6, 7])],
        "exp_power_law": [torch.tensor([0., 0., 0., 1.]), torch.tensor([1e6, 1e6, 1e6, 7])],
        "exp_linear2": [
            torch.tensor([-6.7055, -2.4018, -3.9351]),
            torch.tensor([-2.7502, -2.0065, -3.7675])
        ]
    }

    return params[model_name], empirical_models[model_name], empirical_models_solve[model_name], \
           [params_lbs_ubs[model_name][0].to(device), params_lbs_ubs[model_name][1].to(device)]
