% Sample a sequence from Model 2 with replica cSMC
% and approximate predictive.

function x_new  = pg_m2_approx(y, x_cur, x_tilde_cur, M, ...
    L_post_init, L_post, L_init, L, phi_vec, sig_vec, ...
    mean_1_mat, mean_i_mat)
    
    x_d = size(x_cur, 1);
    n = size(x_cur, 2);
    K_c = size(x_tilde_cur, 3);
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
    lw_tmp = zeros(M, K_c);
    bwd_hat = zeros(M, n);
    mean_mat = zeros(x_d, M, n);
                
    C = L*L';
    C_init = L_init*L_init';
    Phi = diag(phi_vec);
    L_wgt = chol(C + Phi^2\C, 'lower');
    L_wgt_init = chol(C_init + Phi^2\C, 'lower');
    
    l = randi(K_c, M-1, 1);
    l_all = [randi(K_c); l];
    
    x_pool(:, 1, 1) = x_cur(:, 1);
    x_mean = mean_1_mat*reshape(x_tilde_cur(:, 2, l), x_d, M-1);    
    x_pool(:, 2:end, 1) = x_mean + L_post_init*randn(x_d, M-1);  

    mean_mat(:, :, 1) = bsxfun(@times, phi_vec', x_pool(:, :, 1));
    
    for j = 1 : K_c
            
    	lw_tmp(:, j) = mvn_lpdf_L(x_tilde_cur(:, 2, j), mean_mat(:, :, 1), L);
        
    end
    
    bwd_hat(:, 2) = add_logs_mat(lw_tmp)';

    lw(:, 1) = lpois_2_vec(x_pool(:, :, 1), y(:, 1), sig_vec) + ...
        mvn_lpdf_L(Phi\reshape(x_tilde_cur(:, 2, l_all), x_d, M), zeros(x_d, 1), ...
            L_wgt_init)';
            
    w = exp(lw(:, 1) - max(lw(:, 1)));
            
    for i = 2 : n - 1
                               
        l = randi(K_c, M-1, 1);
        l_all = [randi(K_c); l];

        mix_ind_vec = randsample(M, M-1, 'true', w);
           
        x_pool(:, 1, i) = x_cur(:, i);
        x_mean = mean_i_mat*(x_pool(:, mix_ind_vec, i-1) + ...
            reshape(x_tilde_cur(:, i+1, l), x_d, M-1));
        x_pool(:, 2:end, i) = x_mean + L_post*randn(x_d, M-1);

        mean_mat(:, :, i) = bsxfun(@times, phi_vec', x_pool(:, :, i));        
        
        all_past_ind = [1; mix_ind_vec];
        mu_wgt = mean_mat(:, all_past_ind, i-1);
        
        for j = 1 : K_c
            
            lw_tmp(:, j) = mvn_lpdf_L(x_tilde_cur(:, i+1, j), mean_mat(:, :, i), L);
                        
        end
                
        bwd_hat(:, i+1) = add_logs_mat(lw_tmp);
                                 
        lw(:, i) = lpois_2_vec(x_pool(:, :, i), y(:, i), sig_vec) - ...
            bwd_hat(all_past_ind, i)' + ...
            mvn_lpdf_L(Phi\reshape(x_tilde_cur(:, i+1, l_all), x_d, M), ...
            mu_wgt, L_wgt)';
        
        w = exp(lw(:, i) - max(lw(:, i)));
        
    end
        
    x_pool(:, 1, n) = x_cur(:, n);
        
    mix_ind_vec = randsample(M, M-1, 'true', w);

    x_mean = bsxfun(@times, phi_vec', x_pool(:, mix_ind_vec, n-1));
    
    x_prop_all = x_mean + L*randn(x_d, M-1);
    x_pool(:, 2:end, n) = x_prop_all;
    
    all_past_ind = [1; mix_ind_vec];
    
    lw(:, n) = lpois_2_vec(x_pool(:, :, n), y(:, n), sig_vec) - ...
        bwd_hat(all_past_ind, n)';
    
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