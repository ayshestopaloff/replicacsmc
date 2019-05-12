% Sample a sequence from Lorenz-96 model with replica cSMC
% and approximate predictive.

function x_new = pg_lorenz_approx(Observation, Lorenz, interval, ...
    x_cur, x_tilde_cur, M, P_init_obs, P_tr_obs, ...
    L_init_rep, L_rep, L, L_s, L_fwd, L_wgt_init, L_wgt)
    
    x_d = size(x_cur, 1);
    n = size(x_cur, 2);
    obs = x_d-2;
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
    bwd_hat = zeros(M, n);
    mean_mat = zeros(x_d, M, n);
    
    x_mean_init = zeros(x_d, 1);
    
    x_bwd = RungeKutta4(x_tilde_cur(:, 2)',-1,interval,Lorenz)';
    
    x_mean_init(1:obs, :) = (P_init_obs + P_tr_obs)\(P_tr_obs*x_bwd(1:obs));
    x_mean_init(x_d-1:end) = 0;
                    
    x_pool(:, 1, 1) = x_cur(:, 1);
    x_pool(:, 2:end, 1) = x_mean_init + L_init_rep*randn(x_d, M-1);

    mean_mat(:, :, 1) = RungeKutta4(x_pool(:, :, 1)',1,interval,Lorenz)';
    
    bwd_hat(:, 2) = mvn_lpdf_L(x_tilde_cur(1:obs, 2), mean_mat(1:obs, :, 1), L_fwd)';
        
    lw(:, 1) = mvn_lpdf_L(Observation.y(1, 1:obs)', x_pool(1:obs, :, 1), L_s) + ...
        mvn_lpdf_L(x_bwd(1:obs), zeros(obs, 1), L_wgt_init);
           
    w = exp(lw(:, 1) - max(lw(:, 1)));
                      
    for i = 2 : n - 1

        mix_ind_vec = randsample(M, M-1, 'true', w);
        
        x_mean = zeros(x_d, M-1);
        
        x_fwd = mean_mat(:, mix_ind_vec, i-1);
        x_bwd = RungeKutta4(x_tilde_cur(:, i+1)',-1,interval,Lorenz)';                    
        
        x_mean(1:obs, :) = (2*P_tr_obs)\(P_tr_obs*(x_bwd(1:obs) + ...
            x_fwd(1:obs, :)));      
        x_mean(obs+1:end, :) = x_fwd(obs+1:end, :);
           
        x_pool(:, 1, i) = x_cur(:, i);
        x_pool(:, 2:end, i) = x_mean + L_rep*randn(x_d, M-1);
                   
        mean_mat(:, :, i) = RungeKutta4(x_pool(:, :, i)',1,interval,Lorenz)';        
        
        all_past_ind = [1; mix_ind_vec];
        mu_wgt = mean_mat(:, all_past_ind, i-1);

        bwd_hat(:, i+1) = mvn_lpdf_L(x_tilde_cur(1:obs, i+1), ...
                mean_mat(1:obs, :, i), L_fwd)';
                                                 
        lw(:, i) = mvn_lpdf_L(Observation.y(i, 1:obs)', ...
                   	x_pool(1:obs, :, i), L_s) + ...
                   mvn_lpdf_L(x_tilde_cur(1:obs, i+1), ...
                    mean_mat(1:obs, :, i), L_fwd) - ...
                   mvn_lpdf_L(x_pool(1:obs, :, i), ...
                    x_bwd(1:obs, :), L_fwd) + ...
                   mvn_lpdf_L(x_bwd(1:obs), mu_wgt(1:obs, :), ...
                    L_wgt) - bwd_hat(all_past_ind, i);
                        
        w = exp(lw(:, i) - max(lw(:, i)));
        
    end
    
    mix_ind_vec = randsample(M, M-1, 'true', w);
    all_past_ind = [1; mix_ind_vec];
        
    x_mean = mean_mat(:, mix_ind_vec, n-1);
    
    x_pool(:, 1, n) = x_cur(:, n);
    x_pool(:, 2:end, n) = x_mean + L*randn(x_d, M-1);
                                     
    lw(:, n) = mvn_lpdf_L(Observation.y(n, 1:obs)', x_pool(1:obs, :, n), L_s) - ...
        bwd_hat(all_past_ind, n);  
    
    % Sample a new sequence
    
    ind_set = 1:M;
    
    x_new = zeros(x_d, n);

    % Sample the index of the last hidden state
    
    hid_lpr = lw(:, n);
    hid_pr = exp(hid_lpr - max(hid_lpr));
    
    x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);
    x_new(:, n) = x_pool(:, x_ind, n);
        
    % Sample the indices of the remaining states, going backward in time
        
    for i = n - 1 : -1 : 1
                
        x_phi = mean_mat(:, :, i);
        
        hid_lpr = mvn_lpdf_L(x_new(:, i+1), x_phi, L) + lw(:, i) - ...
            bwd_hat(:, i+1);
        hid_pr = exp(hid_lpr - max(hid_lpr));
                        
        x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);       
        x_new(:, i) = x_pool(:, x_ind, i);
        
    end
    
end