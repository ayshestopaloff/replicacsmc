% Experiment 3 for the linear Gaussian model, verify we can achieve 
% fixed precision with fewer particles, standard iterated cSMC.

clear;

% Sample data

load lg_5_250.mat;

% Number of particles

M = 700;

% Number of updates

numiter = 2500;

% Some stored computations

met_precomp;

% Seeds for the different sampler runs

seeds = 1:20;

for j = seeds(1):seeds(end)

    % Set random number generator seed
    
    run_seed = RandStream('mt19937ar','Seed', j);
    RandStream.setGlobalStream(run_seed);
    
    % Set initial sequence values
            
    x_seq_cur = pg_lg_init(y, M, L_init, L, phi_vec, sigma_2);
                
    % Preallocate matrix used to store MCMC samples
    
    x_sample_vec = zeros(numiter+1, x_d, n);
  
    % Store the initial sequence values
    
    x_sample_vec(1, :, :) = x_seq_cur;
    
    % Uncomment to show current and true latent sequences along x_d_plot
    
%     x_d_plot = 1;
%     plot(x(x_d_plot, :), 'Color', [0, 0.5, 0], 'LineWidth', 4); hold on;
%     h1 = plot(x_seq_cur(1, :), 'red');
%     bd = [-20, 20];
%     ylim(bd);
    
    tic;
    for i = 1 : numiter
                                                
        x_seq_cur = pg_lg_prior(y, x_seq_cur, M, L_init, L, ...
                phi_vec, sigma_2);
                                
        x_sample_vec(i+1, :, :) = x_seq_cur;
      
        % Uncomment to show current and true latent sequences along x_d_plot
         
%         delete(h1);
%         h1 = plot(x_seq_cur(1, :),'red'); drawnow;
%         ylim(bd);
        
        disp(i);
        
    end
    elapsed = toc;
    
    % Store the output from the sampler
    
    savefile = strcat('lg_approx_2_prior_2_',num2str(j), '.mat');    
    save(savefile, '-v7.3', 'x_sample_vec', 'x', 'y', ...
         'phi_vec', 'rho', 'M', 'run_seed');
    
end
