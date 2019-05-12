% Poisson observation density for Model 2, 
% vectorized over multiple pool states.

function lprob = lpois_2_vec(x_i, y_i, sig_vec)
               
    lambda = sig_vec(1)*abs(x_i);
        
    lprob = sum(-lambda + bsxfun(@times, y_i, log(lambda)));
       
end