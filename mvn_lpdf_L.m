% Multivariate normal density, where L is lower 
% Cholesky decomposition of covariance matrix.

function lpdf = mvn_lpdf_L(x, Mu, L)
    
    x_shift = bsxfun(@minus, x, Mu);    
    L_x_shift = (L\x_shift);
    log_det_Sigma = 2*sum(log(diag(L)));
    qf = sum(L_x_shift'.^2, 2);
    lpdf = -(1/2)*log_det_Sigma - (1/2)*qf;

end
