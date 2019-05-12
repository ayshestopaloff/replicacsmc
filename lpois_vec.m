% Poisson observation density for Model 1, 
% vectorized over multiple pool states.

function lprob = lpois_vec(x_i, y_i, c_vec, sig_vec)
               
    lambda = exp(bsxfun(@plus, c_vec', bsxfun(@times, sig_vec', x_i)));
        
    lprob = sum(-lambda + bsxfun(@times, y_i.*sig_vec', x_i));

end
