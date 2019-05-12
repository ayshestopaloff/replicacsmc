% Produce figures for Lorenz-96 experiment.

load lorenz_2_prior_1.mat;
figure; plot(x_sample_vec(2:30:end, 1, 45), 'blue.','MarkerSize', 10);
set(gca, 'FontSize', 18);
ylim([-0.35, -0.1])

load lorenz_2_1.mat;
figure; plot(x_sample_vec(2:30:end, 1, 45), 'blue.','MarkerSize', 10);
set(gca, 'FontSize', 18);
ylim([-0.35, -0.1])