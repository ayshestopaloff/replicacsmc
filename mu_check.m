%%% Experiment 1 %%%
 
% To check agreement between mean of posterior sample and KF
% smoother mean, set method to 'lg_approx_2_check'.
%
% Set numruns = 10.

%%% Experiment 3 %%%

% To compute standard error of x_{1,1} for standard iterated 
% cSMC, set method to 'lg_approx_2_prior_2'.
% 
% To compute standard error of x_{1,1} for replica cSMC, set 
% method to 'lg_approx_2_check_2'.
%
% Set numruns = 20.

method = 'lg_approx_2_check_2';
numruns = 20;
burnin_frac = 0;

filename = cell(1, numruns);
filename{1} = strcat(method,'_',num2str(1),'.mat');
load(filename{1});

numiter = size(x_sample_vec, 1);
x_d = size(x_sample_vec, 2);
n = size(x_sample_vec, 3);

mean_all = zeros(x_d, n, numruns);

% Compute the number of iterations to throw away as burnin

burnin = burnin_frac*(numiter-1);

% Get the mean of the samples over the five runs

mean_all(:, :, 1) = squeeze(mean(x_sample_vec(burnin+2:end, :, :), 1));

for i = 2 : numruns

    filename{i} = strcat(method,'_',num2str(i),'.mat');
    load(filename{i});

    mean_all(:, :, i) = squeeze(mean(x_sample_vec(burnin+2:end, :, :), 1));
    
end

mu_centre = mean(mean_all, 3);
mu_se = std(mean_all, 0, 3) / sqrt(numruns);

upper_bound = mu_centre + 2*mu_se;
lower_bound = mu_centre - 2*mu_se;

load lg_5_250.mat;
met_precomp;

[mu_pred, C_pred, mu_upd, C_upd, mu_smooth, C_smooth] = ...
    kf(y, phi_vec, C_init, C, sigma_2);

disp('Proportion of values which agree with KF smoother');
disp(sum(sum(upper_bound - mu_smooth > 0 & mu_smooth - lower_bound > 0))/ ...
    numel(mu_smooth));

disp('Standard error of x_{1,1}');
disp(mu_se(1, 1));