% Run replica cSMC on Model 2, approximate predictive.

clear;

% Sample data

load abs_15_500.mat;

% Number of particles

M = 100;
K = 50;
K_seq = 1:K;

% Number of updates

numiter = 2000;

% Some stored computations

met_precomp;

% Seeds for the different sampler runs

seeds = 1:1;

for j = seeds(1):seeds(end)

    % Set random number generator seed
    
    run_seed = RandStream('mt19937ar','Seed', j);
    RandStream.setGlobalStream(run_seed);
    
    % Set initial sequence values
    
    x_cur_mat = ones(x_d, n, K);
    
    for k = 1 : K

        x_cur_mat(:, :, k) = pg_2_init(y, 10*M, L_init, L, phi_vec, sig_vec);

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

        for l = 1 : 1
            
            x_seq_cur = x_cur_mat(:, :, l);
            
            x_tilde_cur = x_cur_mat(:, :, K_seq(K_seq~=l));
                        
            x_seq_cur = pg_m2_approx(y, x_seq_cur, x_tilde_cur, M, ...
                L_post_init, L_post, L_init, L, phi_vec, sig_vec, ...
                mean_1_mat, mean_i_mat);
            
            x_cur_mat(:, :, l) = x_seq_cur;
            
        end
        
        for l = 2 : K
            
            x_seq_cur = x_cur_mat(:, :, l);
            
            x_tilde_cur = x_cur_mat(:, :, K_seq(K_seq~=l));
                        
            x_seq_cur = pg_m2_prior(y, x_seq_cur, M, L_init, L, ...
                phi_vec, sig_vec);
            
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
    
    savefile = strcat('m2_50_',num2str(j), '.mat');    
    save(savefile, '-v7.3', 'x_sample_vec', 'x', 'y', ...
         'phi_vec', 'rho', 'M', 'run_seed');
    
end
