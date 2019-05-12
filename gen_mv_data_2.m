% Generate sample data from Model 2.

clear;

% Set random number generator seed

data_seed = RandStream('mt19937ar','Seed', 0);
RandStream.setGlobalStream(data_seed);

% Set length of series

n = 500;

% Set latent state space dimension

x_d = 15;

% x - vector of latent variables, y - vector of observations

x = zeros(x_d, n);
y = zeros(x_d, n);

% Set latent process and observation process parameters

phi_vec = 0.9*ones(1, x_d);
sig_vec = 0.8*ones(1, x_d);

% Construct covariance matrix for latent process at times 2 to n.

rho = 0.7;
C = rho*ones(x_d, x_d);
C(1:x_d+1:end) = 1;

% Construct covariance matrix for latent process at time 1

C_init = C.*((1./sqrt(1 - phi_vec.^2))'*(1./sqrt(1 - phi_vec.^2)));

% Compute Cholesky decompositions of covariance matrices

L_init = chol(C_init, 'lower');
L = chol(C, 'lower');

% Generate latent and observed values at time 1

x(:, 1) = L_init*randn(x_d, 1);
y(:, 1) = poissrnd(sig_vec'.*abs(x(:, 1)), x_d, 1);

% Generate latent and observed values at times 2 to n

for i = 2 : n
    
    x(:, i) = phi_vec'.*x(:, i-1) + L*randn(x_d, 1);
    y(:, i) = poissrnd(sig_vec'.*abs(x(:, i)), x_d, 1);
    
end

% Reversed sequence

y_rev = y(:, end:-1:1);

save(strcat('abs_',num2str(x_d),'_', num2str(n), '.mat'));
