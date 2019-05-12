% Compute and plot autocorrelation times.

clear;

% Choose the runs to compute autocorrelation times

% Linear Gaussian experiment 2, exact predictive, 
% 2 replicas, set method = 'lg_exact_2'.
% Linear Gaussian experiment 2, exact predictive, 
% 75 replicas, set method = 'lg_exact_75'.
% Linear Gaussian experiment 2, approx. predictive, 
% 75 replicas, set method = 'lg_approx_75'.
% Linear Gaussian experiment 4, set method = 'lg_approx_2_check_long'.

% Model 1 experiment, set method = 'm1_approx_5'.

method = 'm1_approx_5';

% The time cost of a single sample.

% For all linear Gaussian examples, set to 1.
% For replica cSMC on Model 1 set to 0.80.

sample_cost = 1;

% The number of runs to be used for the autocorrelation time computation

numruns = 5;

% Proportion of the sample to be used as burnin

burnin_frac = 0.1;

% Generate the list of filenames where samples are stored

filename = cell(1, numruns);
filename{1} = strcat(method,'_',num2str(1),'.mat');
load(filename{1});

% Get the dimensions of the matrix storing the samples

numiter = size(x_sample_vec, 1);
x_d = size(x_sample_vec, 2);
n = size(x_sample_vec, 3);

% Compute the number of iterations to throw away as burnin

burnin = burnin_frac*(numiter-1);

% Get the mean of the samples over the five runs

x_sample_mean = x_sample_vec(burnin+2:end, :, :);

for i = 2 : numruns
    
    filename{i} = strcat(method,'_',num2str(i),'.mat');
    
    clear x_sample_vec;
    load(filename{i});
    
    x_sample_mean = x_sample_mean + x_sample_vec(burnin+2:end, :, :);
    
end

x_sample_mean = x_sample_mean/numruns;

% Compute the autocovariances using mean over the five runs

acv_mean = zeros(numiter-burnin-1, x_d, n);

for i = 1 : numruns
    
    clear x_sample_vec;
    load(filename{i});
    
    for j = 1 : x_d
        
        for k = 1 : n
    
            acv_mean(:, j, k) = acv_mean(:, j, k) + ...
                autocov(x_sample_vec(burnin+2:end, j, k), ...
                    x_sample_mean(:, j, k));
            disp(k);    
            
        end
        
    end
    
end

% Average the autocovariances

acv_mean = acv_mean/numruns;

% Compute the autocorrelations

acf_mat = bsxfun(@rdivide, acv_mean, acv_mean(1, :, :));
tau_est = squeeze(1 + 2*sum(acf_mat(1:200, :, :), 1));

% Plot the autocorrelations, adjusted for computation time

plot(sample_cost*tau_est');
set(gca, 'FontSize', 18);