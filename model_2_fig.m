% Produce figures for Model 2 experiment.

load m2_50_1.mat;

figure; plot(x_sample_vec(2:2:end, 1, 300),'blue.','MarkerSize', 10);
ylim([-6, 6]);
set(gca, 'FontSize', 18);

figure; plot(x_sample_vec(2:2:end, 3, 208).*x_sample_vec(2:2:end, 4, 208), ...
    'blue.', 'MarkerSize', 10);
ylim([-10, 15]);
set(gca, 'FontSize', 18);
