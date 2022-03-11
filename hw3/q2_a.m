% Matric No. A0243230X
% Choose class 0 and class 3
clc;
clear;
close all;
lambda = 0;
stddev = 100;

load('mnist_m.mat');
trainIdx = 1:1000; trainIdx = trainIdx';
trainLabel = train_classlabel(trainIdx); trainLabel = double(trainLabel');
trainData = train_data(:, trainIdx); trainData = double(trainData');
for i = 1:1000
    if trainLabel(i)==1||trainLabel(i)==3
        trainLabel(i) = 1;
    else
        trainLabel(i) = 0;
    end
end
testIdx = 1:250; testIdx = testIdx';
testLabel = test_classlabel(testIdx); testLabel = double(testLabel');
testData = test_data(:, testIdx); testData = double(testData');
for i = 1:250
    if testLabel(i)==1||testLabel(i)==3
        testLabel(i) = double(1);
    else
        testLabel(i) = double(0);
    end
end
n_train = length(trainLabel);
n = length(testLabel);

for lambda = [0 0.00001 0.0001 0.01 0.1 1 10 100]
    r_train = dist(trainData');
    phi_train = exp(-r_train.^2/(2*stddev^2));
    w = pinv(phi_train'*phi_train + lambda*eye(n_train)) * phi_train'*trainLabel;

    r = dist(testData, trainData');
    phi = exp(-r.^2/(2*stddev^2));
    testPred = phi * w;
    trainPred = phi_train * w;
    loss = mse(testPred, testLabel);
    loss_train = mse(trainPred, trainLabel);

    trainAcc = zeros(1, 1000);
    testAcc = zeros(1, 1000);
    thr = zeros(1, 1000);
    for i = 1: 1000
        t = (max(trainPred)-min(trainPred)) * (i-1)/1000 + min(trainPred);
        thr(i) = t;
        trainAcc(i) = (sum(trainLabel(trainPred<t)==0) + sum(trainLabel(trainPred>=t)==1)) / n_train;
        testAcc(i) = (sum(testLabel(testPred<t)==0) + sum(testLabel(testPred>=t)==1)) / n;
    end
    figure();
    plot(thr, trainAcc, '.-', thr, testAcc, '^-');
    ylim([0.45 1]); grid on;
    h1 = legend('Train','Test');
    h2 = xlabel('Threshold'); h3 = ylabel('Accuracy');
    h4 = title(sprintf('lambda = %g', lambda));
    set([h1 h2 h3 h4], 'Interpreter', 'latex');
    fprintf('Lambda: %g\nLoss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\n\n', ...
            lambda, loss_train, loss, 100*max(trainAcc), 100*max(testAcc));
end