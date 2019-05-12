% Run replica cSMC on Lorenz-96 model, approximate predictive.

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

M = 200;
K = 2;
K_seq = 1:K;

% Number of updates

numiter = 30000;

% Seeds for the different sampler runs

seeds = 1:1;

for j = seeds(1):seeds(end)

    % Set random number generator seed
    
    run_seed = RandStream('mt19937ar','Seed', j);
    RandStream.setGlobalStream(run_seed);
    
    % Set initial sequence values
    
    x_cur_mat = zeros(x_d, n, K);
    
    for k = 1 : K

        x_cur_mat(:, :, k) = pg_lorenz_init(Initial, Observation, Lorenz, ...
                interval, 3000, L_init, L, L_s);

    end
      
    % Preallocate matrix used to store MCMC samples
    
    x_sample_vec = zeros(numiter+1, x_d, n);
  
    % Store the initial sequence values
    
    x_sample_vec(1, :, :) = x_cur_mat(:, :, 1);
    
      % Uncomment to show current and true latent sequences along x_d_plot
    
%     x_d_plot = 1;
%     plot(x(x_d_plot, :), 'Color', [0, 0.5, 0], 'LineWidth', 4); hold on;
%     h1 = plot(x_cur_mat(x_d_plot, :, 1), 'red');
%     bd = [-20, 20];
%     ylim(bd);
    
    tic;
    for i = 1 : numiter

        for l = 1 : K
            
            x_seq_cur = x_cur_mat(:, :, l);
            
            x_tilde_cur = x_cur_mat(:, :, K_seq(K_seq~=l));
                        
            x_seq_cur = pg_lorenz_approx(Observation, Lorenz, interval, ...
                x_seq_cur, x_tilde_cur, M, P_init_obs, P_tr_obs, ...
                L_init_rep, L_rep, L, L_s, L_fwd, L_wgt_init, L_wgt);
                        
            x_cur_mat(:, :, l) = x_seq_cur;
            
        end
        
        x_sample_vec(i+1, :, :) = x_cur_mat(:, :, 1);
      
        % Uncomment to show current and true latent sequences along x_d_plot
         
%         delete(h1);
%         h1 = plot(x_cur_mat(x_d_plot, :, 1),'red'); drawnow;
%         ylim(bd);
        
        disp(i);
        
    end
    elapsed = toc;
    
    % Store the output from the sampler
    
    savefile = strcat('lorenz_2_',num2str(j), '.mat');    
    save(savefile, '-v7.3', 'x_sample_vec', 'x', ...
        'Observation', 'M', 'run_seed');
    
end
