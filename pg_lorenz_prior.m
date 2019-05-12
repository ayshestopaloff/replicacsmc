% Sample a sequence from Lorenz-96 model with iterated cSMC.

function x_new = pg_lorenz_prior(Observation, Lorenz, interval, x_cur, M, ...
    L_init, L, L_s)

    x_d = size(x_cur, 1);
    n = size(x_cur, 2);
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
    mean_mat = zeros(x_d, M, n);
                
    % Generate set of particles
        
    x_pool(:, 1, 1) = x_cur(:, 1);
    x_pool(:, 2:end, 1) = L_init*randn(x_d, M-1);
    
    mean_fwd = RungeKutta4(x_pool(:, :, 1)',1,interval,Lorenz)';
    mean_mat(:, :, 1) = mean_fwd;    

    lw(:, 1) = mvn_lpdf_L(Observation.y(1, 1:x_d-2)', ...
        x_pool(1:x_d-2, :, 1), L_s);
    
    w = exp(lw(:, 1) - max(lw(:, 1)));
                    
    for i = 2 : n
        
        mix_ind_vec = randsample(M, M-1, 'true', w);
        
        x_fwd = mean_fwd(:, mix_ind_vec);
                        
        x_pool(:, 1, i) = x_cur(:, i);
        x_pool(:, 2:end, i) = x_fwd + L*randn(x_d, M-1);
        
        mean_fwd = RungeKutta4(x_pool(:, :, i)',1,interval,Lorenz)';
        mean_mat(:, :, i) = mean_fwd;
        
        lw(:, i) = mvn_lpdf_L(Observation.y(i, 1:x_d-2)', ...
            x_pool(1:x_d-2, :, i), L_s);
        
        w = exp(lw(:, i) - max(lw(:, i)));
                                                
    end
    
    % Sample a new sequence
    
    ind_set = 1:M;
    
    x_new = zeros(x_d, n);

    % Sample the index of the last hidden state
    
    hid_pr = exp(lw(:, n) - max(lw(:, n)));
    
    x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);
    x_new(:, n) = x_pool(:, x_ind, n);
    
    % Sample the indices of the remaining states, going backward in time
    
    for i = n - 1 : -1 : 1
        
        x_phi = mean_mat(:, :, i);      
        
        hid_lpr = mvn_lpdf_L(x_new(:, i+1), x_phi, L) + lw(:, i);         
        hid_pr = exp(hid_lpr - max(hid_lpr));
                
        x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);       
        x_new(:, i) = x_pool(:, x_ind, i);
        
    end
    
end
