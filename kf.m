% Compute predictive, filtering and smoothing densities with a 
% Kalman filter for the linear Gaussian model.

function [mu_pred, C_pred, mu_upd, C_upd, mu_smooth, C_smooth] = ...
    kf(y, phi_vec, C_init, C, sigma_2)

    x_d = size(y, 1);
    n = size(y, 2);
    
    mu_upd = zeros(x_d, n);
    C_upd = cell(1, n);

    mu_pred = zeros(x_d, n);
    C_pred = cell(1, n);
    
    Phi = diag(phi_vec);
    
    mu_pred(:, 1) = zeros(x_d, 1);
    C_pred{1} = C_init;
    C_obs = sigma_2*eye(x_d);
    
    for i = 1 : n - 1
        
        G = C_pred{i}/(C_pred{i} + C_obs);
        mu_upd(:, i) = mu_pred(:, i) + G*(y(:, i) - mu_pred(:, i));
        C_upd{i} = C_pred{i} - G*C_pred{i};
        
        mu_pred(:, i+1) = Phi*mu_upd(:, i);
        C_pred{i+1} =  Phi*C_upd{i}*Phi' + C;
        
    end
    
    G = C_pred{n}/(C_pred{n} + C_obs);
    mu_upd(:, n) = mu_pred(:, n) + G*(y(:, n) - mu_pred(:, n));
    C_upd{n} = C_pred{n} - G*C_pred{n};

    mu_smooth = zeros(x_d, n);
    C_smooth = cell(1, n);

    mu_smooth(:, n) = mu_upd(:, n);
    C_smooth{n} = C_upd{n};

    for i = n - 1 : - 1 : 1
        
        F = C_upd{i}*(Phi'/C_pred{i+1});
        mu_smooth(:, i) = mu_upd(:, i) + F*(mu_smooth(:, i+1) - ...
            mu_pred(:, i+1));
        C_smooth{i} = F*(C_smooth{i+1} - C_pred{i+1})*F' + C_upd{i};

    end

end

