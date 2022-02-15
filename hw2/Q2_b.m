clear;
close all;
n_hidden = [10];%123456789 10 20 50 100
x       = -3:0.01:3;
x_train = -1.6:0.05:1.6;
x_test  = -1.6:0.01:1.6;
x_show  = -3:0.01:3;
y       = 1.2*sin(pi*x)-cos(2.4*pi*x);
y_train = 1.2*sin(pi*x_train)-cos(2.4*pi*x_train);
y_test  = 1.2*sin(pi*x_test)-cos(2.4*pi*x_test);
fprintf('Staring hiden: %i\n', n_hidden);

net = feedforwardnet(n_hidden, 'trainlm');% 3trainbr, 2trainlm
[net, tr] = train(net, x_train, y_train);
y_predict = net(x_test);
perf = perform(net, y_predict, y_test);
fprintf('Perf: %.4f\n', perf);
y_show = net(x_show);

figure();
hold on;
h1 = plot(x, y, 'g--');
h2 = scatter(x_train, y_train, 'b*');
h3 = plot(x_show, y_show, 'r', 'linewidth', 1.5);
xlabel('x'), ylabel('y');
line([1.6,1.6], [-3,3], 'linestyle', '--'), line([-1.6,-1.6], [-3,3], 'linestyle', '--');
legend([h1, h2, h3], {'Actual Fn', 'Train Set', 'NN Output'});
title('Number of hidden neuron:', n_hidden);
hold off;