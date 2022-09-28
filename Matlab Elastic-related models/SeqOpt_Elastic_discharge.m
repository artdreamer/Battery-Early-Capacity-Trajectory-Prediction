%% Load input-output data
clear all
load processed_dataset_SeqOpt_Elastic_discharge

%% Use the optimal regulation parameter to train and test the model
[minmse, index_minmse] = min(mse_log10a_es);
alpha_optimal = hyper_params(index_minmse,1);
l1_ratio = hyper_params(index_minmse,2);
[B_elasticnet1,FitInfo_elasticnet1] = lasso(x_scale(:, index_x_selected), log10aTrain,...
                    'Alpha',l1_ratio, 'Lambda', alpha_optimal, 'Standardize', false);
beta_opt1 = [B_elasticnet1; FitInfo_elasticnet1.Intercept];
[minmse, index_minmse] = min(mse_b_es);
alpha_optimal = hyper_params(index_minmse,3);
l1_ratio = hyper_params(index_minmse,4);
[B_elasticnet2,FitInfo_elasticnet2] = lasso(x_scale(:, index_x_selected), bTrain,...
                    'Alpha',l1_ratio, 'Lambda', alpha_optimal, 'Standardize', false);
beta_opt2 = [B_elasticnet2; FitInfo_elasticnet2.Intercept];

%% Evaluation
% Prediction accuracy calculation
log10a_train_es = x_scale(:, index_x_selected)*beta_opt1(1:num_features_selected)+beta_opt1(end);
b_train_es = x_scale(:, index_x_selected)*beta_opt2(1:num_features_selected)+beta_opt2(end);
log10_cycle_lives_train_es = (log10(1-abs_capacities(cells_train, num_capacities)./Initial_capacity(cells_train))-log10a_train_es)./b_train_es;
cycle_lives_train_es =  min(max(10.^log10_cycle_lives_train_es, 300), 3000);
log10a_test1_es = x_scale_test1(:, index_x_selected)*beta_opt1(1:num_features_selected)+beta_opt1(end);
b_test1_es = x_scale_test1(:, index_x_selected)*beta_opt2(1:num_features_selected)+beta_opt2(end);
log10_cycle_lives_test1_es = (log10(1-abs_capacities(cells_test1, num_capacities)./Initial_capacity(cells_test1))-log10a_test1_es)./b_test1_es;
cycle_lives_test1_es = min(max(10.^log10_cycle_lives_test1_es, 300), 3000);
log10a_test2_es = x_scale_test2(:, index_x_selected)*beta_opt1(1:num_features_selected)+beta_opt1(end);
b_test2_es = x_scale_test2(:, index_x_selected)*beta_opt2(1:num_features_selected)+beta_opt2(end);
log10_cycle_lives_test2_es = (log10(1-abs_capacities(cells_test2, num_capacities)./Initial_capacity(cells_test2))-log10a_test2_es)./b_test2_es;
cycle_lives_test2_es = min(max(10.^log10_cycle_lives_test2_es, 300), 3000);
log10a_test3_es = x_scale_test3(:, index_x_selected)*beta_opt1(1:num_features_selected)+beta_opt1(end);
b_test3_es = x_scale_test3(:, index_x_selected)*beta_opt2(1:num_features_selected)+beta_opt2(end);
log10_cycle_lives_test3_es = (log10(1-abs_capacities(cells_test3, num_capacities)./Initial_capacity(cells_test3))-log10a_test3_es)./b_test3_es;
cycle_lives_test3_es = min(max(10.^log10_cycle_lives_test3_es, 300), 3000);

% Capacity degradation trajectory prediction error
num_cap = length(num_capacities);
Cap_train = 1 - 10.^(log10a_train_es*ones(1,num_cap)).*(cycle_specific_capacirty(cells_train,num_capacities)).^(b_train_es*ones(1,num_cap));
Cap_test1 = 1 - 10.^(log10a_test1_es*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test1,num_capacities)).^(b_test1_es*ones(1,num_cap));
Cap_test2 = 1 - 10.^(log10a_test2_es*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test2,num_capacities)).^(b_test2_es*ones(1,num_cap));
Cap_test3 = 1 - 10.^(log10a_test3_es*ones(1,num_cap)).*(cycle_specific_capacirty(cells_test3,num_capacities)).^(b_test3_es*ones(1,num_cap));
rmse_train = sqrt(mean(mean((abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train)).^2)));
rmse_test1 = sqrt(mean(mean((abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1)).^2)));
rmse_test2 = sqrt(mean(mean((abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2)).^2)));
rmse_test3 = sqrt(mean(mean((abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3)).^2)));
mae_train = (mean(mean(abs(abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train)))));
mae_test1 = (mean(mean(abs(abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1)))));
mae_test2 = (mean(mean(abs(abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2)))));
mae_test3 = (mean(mean(abs(abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3)))));
mape_train = (mean(mean(abs((abs_capacities(cells_train, num_capacities) - Cap_train.*Initial_capacity(cells_train))./abs_capacities(cells_train, num_capacities)))));
mape_test1 = (mean(mean(abs((abs_capacities(cells_test1, num_capacities) - Cap_test1.*Initial_capacity(cells_test1))./abs_capacities(cells_test1, num_capacities)))));
mape_test2 = (mean(mean(abs((abs_capacities(cells_test2, num_capacities) - Cap_test2.*Initial_capacity(cells_test2))./abs_capacities(cells_test2, num_capacities)))));
mape_test3 = (mean(mean(abs((abs_capacities(cells_test3, num_capacities) - Cap_test3.*Initial_capacity(cells_test3))./abs_capacities(cells_test3, num_capacities)))));

% cycle prediction error
rmse_cycle_train = sqrt(mean(mean((cycle_lives_train_es-cycle_specific_capacirty(cells_train,num_capacities)).^2)));
rmse_cycle_test1 = sqrt(mean(mean((cycle_lives_test1_es-cycle_specific_capacirty(cells_test1,num_capacities)).^2)));
rmse_cycle_test2 = sqrt(mean(mean((cycle_lives_test2_es-cycle_specific_capacirty(cells_test2,num_capacities)).^2)));
rmse_cycle_test3 = sqrt(mean(mean((cycle_lives_test3_es-cycle_specific_capacirty(cells_test3,num_capacities)).^2)));
mpe_cycle_train = mean(abs((cycle_lives_train_es-cycle_specific_capacirty(cells_train,num_capacities))./cycle_specific_capacirty(cells_train,num_capacities)));
map_cycle_test1 = mean(abs((cycle_lives_test1_es-cycle_specific_capacirty(cells_test1,num_capacities))./cycle_specific_capacirty(cells_test1,num_capacities)));
map_cycle_test2 = mean(abs((cycle_lives_test2_es-cycle_specific_capacirty(cells_test2,num_capacities))./cycle_specific_capacirty(cells_test2,num_capacities)));
map_cycle_test3 = mean(abs((cycle_lives_test3_es-cycle_specific_capacirty(cells_test3,num_capacities))./cycle_specific_capacirty(cells_test3,num_capacities)));

%  Cycle life prediction error
log10_cycle_lives_train_es = (log10(1-0.88./Initial_capacity(cells_train))-log10a_train_es)./b_train_es;
cycle_lives_train_es = min(max(10.^log10_cycle_lives_train_es, 300), 3000);
log10_cycle_lives_test1_es = (log10(1-0.88./Initial_capacity(cells_test1))-log10a_test1_es)./b_test1_es;
cycle_lives_test1_es = min(max(10.^log10_cycle_lives_test1_es, 300), 3000);
log10_cycle_lives_test2_es = (log10(1-0.88./Initial_capacity(cells_test2))-log10a_test2_es)./b_test2_es;
cycle_lives_test2_es = min(max(10.^log10_cycle_lives_test2_es, 300), 3000);
log10_cycle_lives_test3_es = (log10(1-0.88./Initial_capacity(cells_test3))-log10a_test3_es)./b_test3_es;
cycle_lives_test3_es = min(max(10.^log10_cycle_lives_test3_es, 300), 3000);
rmse_cycle_train = sqrt(mean(mean((cycle_lives_train_es-cycle_lives(cells_train)).^2)));
rmse_cycle_test1 = sqrt(mean(mean((cycle_lives_test1_es-cycle_lives(cells_test1)).^2)));
rmse_cycle_test2 = sqrt(mean(mean((cycle_lives_test2_es-cycle_lives(cells_test2)).^2)));
rmse_cycle_test3 = sqrt(mean(mean((cycle_lives_test3_es-cycle_lives(cells_test3)).^2)));
mae_cycle_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))));
mae_cycle_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))));
mae_cycle_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))));
mae_cycle_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))));
mape_cycle_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))./cycle_lives(cells_train)));
mape_cycle_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))./cycle_lives(cells_test1)));
mape_cycle_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))./cycle_lives(cells_test2)));
mape_cycle_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))./cycle_lives(cells_test3)));

