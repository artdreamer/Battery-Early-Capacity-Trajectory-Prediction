%% Load input-output data
clear all
load processed_dataset_E2E_Elastic_deltaQ 
num_features_selected = size(x_scale, 2);

%% Set up the capacity degradation model parameter limits
b_average = mean(bTrain);
b_max = max(bTrain) * 1.2;
b_min = min(bTrain) * 0.8;
n_min = 300;
n_max = 3000;
log10_a_min = log10(1-0.8) - b_max*log10(n_max);
log10_a_max = log10(1-0.8) - b_min*log10(n_min);

%% Use the optimal regulation parameter to train and test the model
n_min = 300;
n_max = 3000;
alpha_optimal = 0.160390988559592;
l1_ratio = 0.000658667586112327;
lambda = 0.216711356743881;
num_capacities = 1:100;
[beta_opt, cvx_val, ~, ~] = joint_optimization_log10_cycle(x_scale, log10(cycle_specific_capacirty(cells_train,num_capacities)), cycle_lives(cells_train),...
                Initial_capacity(cells_train), abs_capacities(cells_train, num_capacities)./Initial_capacity(cells_train), alpha_optimal, l1_ratio, lambda, n_max, n_min, log10_a_max, log10_a_min, b_max, ...
                b_min, b_average, false);
            

%% Evaluation
num_capacities = 1:100;
log10_a_bar_train = x_scale*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_train = x_scale*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_train_es = (log10(1-abs_capacities(cells_train, num_capacities)./Initial_capacity(cells_train))-log10_a_bar_train)./b_bar_train;
cycle_train_es = 10.^log10_cycle_train_es;
log10_a_bar_test1 = x_scale_test1*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test1 = x_scale_test1*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_test1_es = (log10(1-abs_capacities(cells_test1, num_capacities)./Initial_capacity(cells_test1))-log10_a_bar_test1)./b_bar_test1;
cycle_test1_es = 10.^log10_cycle_test1_es;
log10_a_bar_test2 = x_scale_test2*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test2 = x_scale_test2*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_test2_es = (log10(1-abs_capacities(cells_test2, num_capacities)./Initial_capacity(cells_test2))-log10_a_bar_test2)./b_bar_test2;
cycle_test2_es = 10.^log10_cycle_test2_es;
log10_a_bar_test3 = x_scale_test3*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test3 = x_scale_test3*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_test3_es = (log10(1-abs_capacities(cells_test3, num_capacities)./Initial_capacity(cells_test3))-log10_a_bar_test3)./b_bar_test3;
cycle_test3_es = 10.^log10_cycle_test3_es;

% Capacity degradation trajectory prediction error 
num_cap = length(num_capacities);
Cap_train = 1 - 10.^(log10_a_bar_train*ones(1,num_cap)).*(cycle_specific_capacirty(cells_train,num_capacities)).^(b_bar_train*ones(1,num_cap));
Cap_test1 = 1 - 10.^(log10_a_bar_test1*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test1,num_capacities)).^(b_bar_test1*ones(1,num_cap));
Cap_test2 = 1 - 10.^(log10_a_bar_test2*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test2,num_capacities)).^(b_bar_test2*ones(1,num_cap));
Cap_test3 = 1 - 10.^(log10_a_bar_test3*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test3,num_capacities)).^(b_bar_test3*ones(1,num_cap));
rmse_train = sqrt(mean(mean((abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train)).^2)));
rmse_test1 = sqrt(mean(mean((abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1)).^2)));
rmse_test2 = sqrt(mean(mean((abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2)).^2)));
rmse_test3 = sqrt(mean(mean((abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3)).^2)));
mae_train = mean(mean(abs(abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train))));
mae_test1 = mean(mean(abs(abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1))));
mae_test2 = mean(mean(abs(abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2))));
mae_test3 = mean(mean(abs(abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3))));
mape_train = mean(mean(abs((abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train))./abs_capacities(cells_train, num_capacities))));
mape_test1 = mean(mean(abs((abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1))./abs_capacities(cells_test1, num_capacities))));
mape_test2 = mean(mean(abs((abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2))./abs_capacities(cells_test2, num_capacities))));
mape_test3 = mean(mean(abs((abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3))./abs_capacities(cells_test3, num_capacities))));

% Cycle prediction error 
rmse_cycle_train = sqrt(mean(mean((cycle_train_es-cycle_specific_capacirty(cells_train,num_capacities)).^2)));
rmse_cycle_test1 = sqrt(mean(mean((cycle_test1_es-cycle_specific_capacirty(cells_test1,num_capacities)).^2)));
rmse_cycle_test2 = sqrt(mean(mean((cycle_test2_es-cycle_specific_capacirty(cells_test2,num_capacities)).^2)));
rmse_cycle_test3 = sqrt(mean(mean((cycle_test3_es-cycle_specific_capacirty(cells_test3,num_capacities)).^2)));
mpe_cycle_train = mean(mean(abs((cycle_train_es-cycle_specific_capacirty(cells_train,num_capacities))./cycle_specific_capacirty(cells_train,num_capacities))));
map_cycle_test1 = mean(mean(abs((cycle_test1_es-cycle_specific_capacirty(cells_test1,num_capacities))./cycle_specific_capacirty(cells_test1,num_capacities))));
map_cycle_test2 = mean(mean(abs((cycle_test2_es-cycle_specific_capacirty(cells_test2,num_capacities))./cycle_specific_capacirty(cells_test2,num_capacities))));
map_cycle_test3 = mean(mean(abs((cycle_test3_es-cycle_specific_capacirty(cells_test3,num_capacities))./cycle_specific_capacirty(cells_test3,num_capacities))));

% Cycle life prediction error (
cap_level = 0.8*1.1;
log10_a_bar_train = x_scale*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_train = x_scale*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_lives_train_es = (log10(1-cap_level./Initial_capacity(cells_train))-log10_a_bar_train)./b_bar_train;
cycle_lives_train_es = 10.^log10_cycle_lives_train_es;
log10_a_bar_test1 = x_scale_test1*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test1 = x_scale_test1*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_lives_test1_es = (log10(1-cap_level./Initial_capacity(cells_test1))-log10_a_bar_test1)./b_bar_test1;
cycle_lives_test1_es = 10.^log10_cycle_lives_test1_es;
log10_a_bar_test2 = x_scale_test2*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test2 = x_scale_test2*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_lives_test2_es = (log10(1-cap_level./Initial_capacity(cells_test2))-log10_a_bar_test2)./b_bar_test2;
cycle_lives_test2_es = 10.^log10_cycle_lives_test2_es;
log10_a_bar_test3 = x_scale_test3*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
b_bar_test3 = x_scale_test3*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
log10_cycle_lives_test3_es = (log10(1-cap_level./Initial_capacity(cells_test3))-log10_a_bar_test3)./b_bar_test3;
cycle_lives_test3_es = 10.^log10_cycle_lives_test3_es;

% Cycle life prediction error
rmse_cyclelife_train = sqrt(mean(mean((cycle_lives_train_es-cycle_lives(cells_train)).^2)));
rmse_cyclelife_test1 = sqrt(mean(mean((cycle_lives_test1_es-cycle_lives(cells_test1)).^2)));
rmse_cyclelife_test2 = sqrt(mean(mean((cycle_lives_test2_es-cycle_lives(cells_test2)).^2)));
rmse_cyclelife_test3 = sqrt(mean(mean((cycle_lives_test3_es-cycle_lives(cells_test3)).^2)));

mae_cyclelife_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))));
mae_cyclelife_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))));
mae_cyclelife_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))));
mae_cyclelife_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))));

mape_cyclelife_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))./cycle_lives(cells_train)));
mape_cyclelife_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))./cycle_lives(cells_test1)));
mape_cyclelife_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))./cycle_lives(cells_test2)));
mape_cyclelife_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))./cycle_lives(cells_test3)));
