% Run SMC and sample a sequence from Model 2.

function x_new = pg_2_init(y, M, L_init, L, phi_vec, sig_vec)

    x_d = size(y, 1);
    n = size(y, 2);
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
    
    % Generate set of particles
        
    x_pool(:, :, 1) = L_init*randn(x_d, M);
    
    lw(:, 1) = lpois_2_vec(x_pool(:, :, 1), y(:, 1), sig_vec);
    w = exp(lw(:, 1) - max(lw(:, 1)));
        
    for i = 2 : n
                
        mix_ind_vec = randsample(M, M, 'true', w);
        
        x_mean = bsxfun(@times, phi_vec', x_pool(:, mix_ind_vec, i-1));

        x_prop_all = x_mean + L*randn(x_d, M);
        x_pool(:, :, i) = x_prop_all;
        
        lw(:, i) = lpois_2_vec(x_pool(:, :, i), y(:, i), sig_vec);
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
                
        x_phi = bsxfun(@times, phi_vec', x_pool(:, :, i));
        
        hid_lpr = mvn_lpdf_L(x_new(:, i+1), x_phi, L) + lw(:, i);         
        hid_pr = exp(hid_lpr - max(hid_lpr));
                
        x_ind = ind_set(sum((rand(1) >= cumsum(hid_pr./sum(hid_pr)))) + 1);       
        x_new(:, i) = x_pool(:, x_ind, i);
        
    end
    
end
