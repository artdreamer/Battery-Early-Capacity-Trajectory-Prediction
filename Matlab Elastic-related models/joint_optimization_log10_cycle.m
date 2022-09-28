function [beta_opt, cvx_optval, mse_log10_cycle, mse_cycle] = joint_optimization_log10_cycle(x, log10_cycle, cycle_lives, Initial_capacity, normalized_capacity, ...
    alpha, l1_ratio, lambda, n_max, n_min, log10_a_max, log10_a_min, b_max, b_min, b0, plot_shown)

% This function is to train the liner regression model for learning power
% law capacity degradation mode parameters. The power-law capacity
% degradation model is C = 1 - a*k^b, where k is the cycle number, C is the
% remaining normalized capacity, a and b are model parameters to be learn
% though the linear regression model.

% Inputs:
%     x: input features
%     log10_cycle: log10 functio of cycles at specific capacities
%     normalized_capacity: normalized specific capacities
%     alpha: regulation magnitude
%     l1_ratio: proportion of li-norm in the regulation
%     n_max: cycle life upper limit
%     n_min: cycle life lower limit
%     log10_a_max: upper limit of log10(a)
%     log10_a_min: lower limit of log10(a)
%     b_max: upper limit of b
%     b_min: lower limit of b
% Outputs:
%     beta_opt: parameters of the linear regression model 
%     mse_log10_cycle: mean squared error for the log10(cycle) prediction (across different datapoints and capacities)
%     mse_cycle: mean squared error for the cycle prediction (across different datapoints and capacities)

    data_size = size(x, 1);
    num_cap = size(normalized_capacity, 2);
    num_features_selected = size(x, 2);
    index_beta_relevant1 = 1:num_features_selected;
    index_beta_relevant2 = num_features_selected+2:num_features_selected*2+1;
    index_beta_relevant = [index_beta_relevant1, index_beta_relevant2];
    %     cvx_solver mosek
%     cvx_begin quiet
    cvx_begin
%     cvx_precision best
        variables beta_joint_3(2*num_features_selected+2, 1)
        expressions error_matrx(data_size, num_cap) obj_log_cycle_life log10_a_bar(data_size, 1) b_bar(data_size, 1) error_vector(data_size, 1)
        log10_a_bar = x*beta_joint_3(1:num_features_selected)+beta_joint_3(num_features_selected+1);
        b_bar = x*beta_joint_3(num_features_selected+2:end-1)+beta_joint_3(end);
%         error_matrx = (log10(1-normalized_capacity) -...
%         log10_a_bar*ones(1,num_cap) - b_bar*ones(1,num_cap).*log10_cycle);
        error_matrx = (log10(1-normalized_capacity) -...
        log10_a_bar*ones(1,num_cap) - b_bar*ones(1,num_cap).*log10_cycle);
        error_vector = (log10(1-0.88./Initial_capacity) -...
        log10_a_bar*ones - b_bar.*log10(cycle_lives));
        obj_log_cycle_life = (1-lambda)*1/2/data_size/num_cap*square_pos(norm(error_matrx, 'fro')) + ...
            lambda*1/2/data_size*square_pos(norm(error_vector, 2))+...
        alpha*l1_ratio*norm(beta_joint_3(index_beta_relevant), 1) +...
        alpha*(1-l1_ratio)/2*square_pos(norm(beta_joint_3(index_beta_relevant)));
        minimize(obj_log_cycle_life)
        subject to    
        %Inequalities
        % cycle life upper and lower limits
        log10(1-0.88./Initial_capacity)-log10_a_bar<=log10(n_max)*b_bar;
        log10(1-0.88./Initial_capacity)-log10_a_bar>=log10(n_min)*b_bar;
        % parameter a and b upper and lower limits
        b_min<=b_bar<=b_max;
        log10_a_min<=log10_a_bar<=log10_a_max;
%         beta_joint_3([10], 1) == 0;
%         beta_joint_3([3], 1) == 0;
%         beta_joint_3(1*num_features_selected+1, 1) == -10.7491;
        beta_joint_3(end) == b0;
%         min(b_bar, 1)/data_size == 3.45;
%         sum(b_bar, 1)/data_size == 4;
%         mean(log10_a_min, 1) == -10.7491;
%         log10(1-0.88./Initial_capacity)-log10_a_bar<=log10(n_max)*b_bar;
    cvx_end
    beta_opt = beta_joint_3;
    log10_a_bar = x*beta_opt(1:num_features_selected)+beta_opt(num_features_selected+1);
    b_bar = x*beta_opt(num_features_selected+2:end-1)+beta_opt(end);
    log10_cycle_lives_es = (log10(1-normalized_capacity)-log10_a_bar)./b_bar;
    cycle_lives_es = 10.^log10_cycle_lives_es;
    mse_log10_cycle = mean(mean((log10_cycle_lives_es-log10_cycle).^2));
    mse_cycle = mean(mean((cycle_lives_es-10.^log10_cycle).^2));
%     cycle_lives_es_stage1
%     mse_cycle1 = mean(mean((cycle_lives_es_stage1-10.^log10_cycle).^2));
%     mse_cycle2 = mean(mean((cycle_lives_es_stage1-cycle_lives_es).^2));
%     mse_cycle1 = mean(mean((cycle_lives_es-10.^log10_cycle).^2));
    if plot_shown
        figure
        plot(10.^log10_cycle, cycle_lives_es,...
        'LineStyle', 'none',...
        'Marker','s',...
        'MarkerIndices',1:data_size,...
        'MarkerFaceColor','none',...
        'MarkerSize', 3,...
        'linewidth',1.5);
        hold on;
        plot(10.^log10_cycle,10.^log10_cycle,...
        'LineStyle', '-',...
        'Color', 'blue',...
        'Marker','none',...
        'MarkerFaceColor','none',...
        'MarkerEdgeColor','blue',...
        'MarkerSize', 3,...
        'linewidth',1.5);
        grid on;
%         axis([335,2500,335,2500]);
        xlabel('True cycle number')
        ylabel('Predicted cycle number')
        set(findall(gcf,'-property','FontSize'),'FontSize',12)
        set(gcf,'Position',[100 100 500 400]);
    end
