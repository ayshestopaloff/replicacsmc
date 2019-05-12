% Run SMC and sample a sequence from the Lorenz-96 model.

function x_new = pg_lorenz_init(Initial, Observation, Lorenz, ...
    interval, M, L_init, L, L_s)

    x_d = length(Initial.Mean);
    n = size(Observation.y, 1);
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
            
    % Generate set of particles
        
    x_pool(:, :, 1) = L_init*randn(x_d, M);

    lw(:, 1) = mvn_lpdf_L(Observation.y(1, 1:x_d-2)', x_pool(1:x_d-2, :, 1), L_s);
    
    w = exp(lw(:, 1) - max(lw(:, 1)));
        
    for i = 2 : n
                        
        mix_ind_vec = randsample(M, M, 'true', w);
        
        x_fwd = RungeKutta4(x_pool(:, mix_ind_vec, i-1)', 1,interval,Lorenz)';
        
        x_pool(:, :, i) = x_fwd + L*randn(x_d, M);
                
        lw(:, i) = mvn_lpdf_L(Observation.y(i, 1:x_d-2)', x_pool(1:x_d-2, :, i), L_s);
        
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
        
        x_phi = RungeKutta4(x_pool(:, :, i)',1,interval,Lorenz)';       
        
        hid_lpr = mvn_lpdf_L(x_new(:, i+1), x_phi, L) + lw(:, i);         
        hid_pr = exp(hid_lpr - max(hid_lpr));
                
        x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);       
        x_new(:, i) = x_pool(:, x_ind, i);
        
    end
    
end
