% Compute log(A_1 + A_2 + ... + A_n) where 
% a_vec = (log(A_1), log(A_2), ... , log(A_n)).

function total = add_logs_mat(a_vec)
    
    a_max = max(a_vec, [], 2);
    total = a_max + log(sum(exp(bsxfun(@minus, a_vec, a_max)), 2));

end