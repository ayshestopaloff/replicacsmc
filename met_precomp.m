% Precomputations for conditional density of 
% x_t given x_t-1 and x_t+1.

Phi = diag(phi_vec);

C_inv = inv(C);
C_init_inv = inv(C_init);
C_lik_inv = Phi*(C\Phi);

L_post_init = chol(inv(C_init_inv + C_lik_inv), 'lower');
L_post = chol(inv(C_inv + C_lik_inv), 'lower');

mean_1_mat = (Phi^2 + C_init\C)\Phi;
mean_i_mat = (Phi^2 + eye(x_d))\Phi;