%% Load input-output data
clear all
load processed_dataset_Elastic_discharge

%% Use the optimal regulation parameter to train and test the model
alpha_optimal = 0.104223938829999;
l1_ratio = 0.0100000000000000;
[B_elasticnet,FitInfo_elasticnet] = lasso(x_scale(:, index_x_selected), log10(cycle_lives(cells_train)),...
                    'Alpha',l1_ratio, 'Lambda', alpha_optimal, 'Standardize', false);
beta_opt = [B_elasticnet; FitInfo_elasticnet.Intercept];

%% Evaluation
% Prediction accuracy calculation
log10_cycle_lives_train_es = x_scale(:, index_x_selected)*beta_opt(1:num_features_selected)+beta_opt(end);
cycle_lives_train_es = 10.^log10_cycle_lives_train_es;
log10_cycle_lives_test1_es = x_scale_test1(:, index_x_selected)*beta_opt(1:num_features_selected)+beta_opt(end);
cycle_lives_test1_es = 10.^log10_cycle_lives_test1_es;
log10_cycle_lives_test2_es = x_scale_test2(:, index_x_selected)*beta_opt(1:num_features_selected)+beta_opt(end);
cycle_lives_test2_es = 10.^log10_cycle_lives_test2_es;
log10_cycle_lives_test3_es = x_scale_test3(:, index_x_selected)*beta_opt(1:num_features_selected)+beta_opt(end);
cycle_lives_test3_es = 10.^log10_cycle_lives_test3_es;
rmse_cycle_train = sqrt(mean((cycle_lives_train_es-cycle_lives(cells_train)).^2));
rmse_cycle_test1 = sqrt(mean((cycle_lives_test1_es-cycle_lives(cells_test1)).^2));
rmse_cycle_test2 = sqrt(mean((cycle_lives_test2_es-cycle_lives(cells_test2)).^2));
rmse_cycle_test3 = sqrt(mean((cycle_lives_test3_es-cycle_lives(cells_test3)).^2));
mae_cycle_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))));
mae_cycle_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))));
mae_cycle_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))));
mae_cycle_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))));
mape_cycle_train = mean(abs((cycle_lives_train_es-cycle_lives(cells_train))./cycle_lives(cells_train)));
mape_cycle_test1 = mean(abs((cycle_lives_test1_es-cycle_lives(cells_test1))./cycle_lives(cells_test1)));
mape_cycle_test2 = mean(abs((cycle_lives_test2_es-cycle_lives(cells_test2))./cycle_lives(cells_test2)));
mape_cycle_test3 = mean(abs((cycle_lives_test3_es-cycle_lives(cells_test3))./cycle_lives(cells_test3)));