% Run standard iterated cSMC on Lorenz-96 model.

clear;

% Sample data

load lorenz_16_100.mat;
n = size(Observation.y', 2);
x = Latentx';
x_d = d;
obs = x_d-2;

L_init = chol(Initial.Cov, 'lower');
L = chol(Transition.G, 'lower');

P_init_obs = Initial.Precision(1:obs, 1:obs);
P_tr_obs = Transition.Ginv(1:obs, 1:obs);

C_init_obs = Initial.Cov(1:obs, 1:obs);
C_tr_obs = Transition.G(1:obs, 1:obs);

L_init_rep = chol(diag(1./[diag(P_init_obs) + diag(P_tr_obs); ...
    diag(Initial.Precision(obs+1:end, obs+1:end))]), 'lower');

L_rep = chol(diag(1./[2*diag(P_tr_obs); ...
    diag(Transition.Ginv(obs+1:end, obs+1:end))]), 'lower');

L_s = chol((Observation.Std^2) * eye(obs), 'lower');
L_fwd = chol(C_tr_obs, 'lower');
L_wgt_init = chol(C_tr_obs + C_init_obs, 'lower');
L_wgt = chol(2*C_tr_obs, 'lower');

% Number of particles

M = 600;

% Number of updates

numiter = 30000;

% Seeds for the different sampler runs

seeds = 1:1;

for j = seeds(1):seeds(end)

    % Set random number generator seed
    
    run_seed = RandStream('mt19937ar','Seed', j);
    RandStream.setGlobalStream(run_seed);
    
    % Set initial sequence values
        
    x_seq_cur = pg_lorenz_init(Initial, Observation, Lorenz, ...
            interval, 3000, L_init, L, L_s);

    % Preallocate matrix used to store MCMC samples
    
    x_sample_vec = zeros(numiter+1, x_d, n);
  
    % Store the initial sequence values
    
    x_sample_vec(1, :, :) = x_seq_cur;
    
      % Uncomment to show current and true latent sequences along x_d_plot
    
%     x_d_plot = 1;
%     plot(x(x_d_plot, :), 'Color', [0, 0.5, 0], 'LineWidth', 4); hold on;
%     h1 = plot(x_seq_cur(x_d_plot, :), 'red');
%     bd = [-20, 20];
%     ylim(bd);
    
    tic;
    for i = 1 : numiter
        
        x_seq_cur = pg_lorenz_prior(Observation, Lorenz, interval, ...
            x_seq_cur, M, L_init, L, L_s);
        
        x_sample_vec(i+1, :, :) = x_seq_cur;
      
        % Uncomment to show current and true latent sequences along x_d_plot
         
%         delete(h1);
%         h1 = plot(x_seq_cur(x_d_plot, :),'red'); drawnow;
%         ylim(bd);
        
        disp(i);
        
    end
    elapsed = toc;
    
    % Store the output from the sampler
    
    savefile = strcat('lorenz_2_prior_',num2str(j), '.mat');    
    save(savefile, '-v7.3', 'x_sample_vec', 'x', ...
        'Observation', 'M', 'run_seed');
    
end
