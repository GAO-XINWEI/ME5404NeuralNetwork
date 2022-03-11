clc;
clear all;
close all;
M = 20;
lambda = 0;
x_test = -1.6:0.01:1.6;
x_train = -1.6:0.08:1.6;
n_train = size(x_train, 2);
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);
y_train = 1.2*sin(pi*x_train) - cos(2.4*pi*x_train) + 0.3*randn(1, n_train);

x_sample = datasample(x_train, M);
dmax = max(x_sample) - min(x_sample);
stddev = dmax / sqrt(2*M);
r_train = x_train' - x_sample;
phi_train = exp(-r_train.^2/(2*stddev^2));
% w = pinv(phi_train) * y_train';
w = pinv(phi_train'*phi_train + lambda*eye(size(phi_train, 2)))*phi_train'*y_train';

r = x_test' - x_sample;
phi = exp(-r.^2/(2*stddev^2));
y_pred = (phi * w)';
loss = mse(y_pred, y_test);
loss_train = mse((phi_train*w)', y_train);
fprintf('Lambda: %.2f\nSigma: %f\nLoss train: %f\nLoss test: %f\n', lambda, stddev, loss_train, loss);

figure();
scatter(x_train, y_train, 'b*');
hold on, plot(x_test, y_test, 'g--', 'linewidth', 1.5);
hold on, plot(x_test, y_pred, 'r', 'linewidth', 1.5);
grid on;
h1 = xlabel('$x$'); h2 = ylabel('$y$');
h3 = legend('Train set', 'Ground truth', 'RBFN output', 'Location', 'northwest');
h4 = title(sprintf('RBFN function approximation with fixed centres. (lambda=%g)', lambda));
set([h1 h2 h3 h4], 'Interpreter', 'latex');

