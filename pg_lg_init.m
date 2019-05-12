% Run SMC and sample a sequence from linear Gaussian model.

function x_new = pg_lg_init(y, M, L_init, L, phi_vec, sigma_2)

    x_d = length(phi_vec);
    n = size(y, 2);
    
    x_pool = zeros(x_d, M, n);
    lw = zeros(M, n);
        
    % Generate set of particles
    
    x_pool(:, :, 1) = L_init*randn(x_d, M);
    
    lw(:, 1) = mvn_lpdf_L(y(:, 1), x_pool(:, :, 1), sqrt(sigma_2)*eye(x_d))';
    w = exp(lw(:, 1) - max(lw(:, 1)));
        
    for i = 2 : n
                
        mix_ind_vec = randsample(M, M, 'true', w);
        
        x_mean = bsxfun(@times, phi_vec', x_pool(:, mix_ind_vec, i-1));
        x_pool(:, :, i) = x_mean + L*randn(x_d, M);
                        
        lw(:, i) = mvn_lpdf_L(y(:, i), x_pool(:, :, i), ...
            sqrt(sigma_2)*eye(x_d))';
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
