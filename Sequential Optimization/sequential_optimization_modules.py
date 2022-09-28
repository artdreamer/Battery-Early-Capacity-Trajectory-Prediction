"""Sequence optimization modules."""
from torch import nn
class SequentialOptimization(nn.Module):
    """Define sequential optimization framework."""
    def __init__(self, input_size, layers, params, empirical_model, params_lbs_ub, act_func,
                 isProb=False):
        super(SequentialOptimization, self).__init__()
        # torch.set_default_dtype(torch.float64)
        self.flatten = nn.Flatten()
        self.empirical_model = empirical_model
        self.params_len = len(params)
        self.params_lbs, self.params_ubs = params_lbs_ub[0], params_lbs_ub[1]
        self.constrain = nn.Sigmoid()

        self.linear_act_func_stack = nn.Sequential()
        for num_layper, num_neurons in enumerate(layers):
            output_size = num_neurons
            self.linear_act_func_stack.add_module(
                "layer " + str(num_layper),
                nn.Sequential(nn.Linear(input_size, output_size), act_func)
            )
            input_size = output_size

        if not isProb:
            self.linear_act_func_stack.add_module(
                "output layer", nn.Linear(input_size, len(params))
            )

    def forward(self, x):
        x = self.flatten(x)
        unconstrained_empirical_model_params = self.linear_act_func_stack(x)
        constrained_empirical_model_params = (self.params_ubs - self.params_lbs) * \
                                             self.constrain(unconstrained_empirical_model_params) \
                                             + self.params_lbs
        return constrained_empirical_model_params
