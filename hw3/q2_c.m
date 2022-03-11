% Matric No. A0243230X
% Choose class 0 and class 3
clc;
clear;
lambda = 0;
n_center = 2;
stddev = 5000;

%% Data init
load('mnist_m.mat');
trainIdx = 1:1000; trainIdx = trainIdx';
trainLabel = train_classlabel(trainIdx); trainLabel = double(trainLabel');
trainData = train_data(:, trainIdx); trainData = double(trainData');
for i = 1:1000
    if trainLabel(i)==0||trainLabel(i)==3
        trainLabel(i) = 1;
    else
        trainLabel(i) = 0;
    end
end
testIdx = 1:250; testIdx = testIdx';
testLabel = test_classlabel(testIdx); testLabel = double(testLabel');
testData = test_data(:, testIdx); testData = double(testData');
for i = 1:250
    if testLabel(i)==0||testLabel(i)==3
        testLabel(i) = 1;
    else
        testLabel(i) = 0;
    end
end
n_train = length(trainLabel);
n_test = length(testLabel);

%% Cluster
[idx, center] = kmeans(trainData,n_center);

%% RBFN
r_train = dist(trainData, center');
phi_train = exp(-r_train.^2/(2*stddev^2));
w = pinv(phi_train'*phi_train + lambda*eye(n_center)) * phi_train'*trainLabel;

r = dist(testData, center');
phi = exp(-r.^2/(2*stddev^2));
testPred = phi * w;
trainPred = phi_train * w;
loss = mse(testPred, testLabel);
loss_train = mse(trainPred, trainLabel);

%% Performance and plot
trainAcc = zeros(1, 1000);
testAcc = zeros(1, 1000);
thr = zeros(1, 1000);
for i = 1: 1000
    t = (max(trainPred)-min(trainPred)) * (i-1)/1000 + min(trainPred);
    thr(i) = t;
    trainAcc(i) = (sum(trainLabel(trainPred<t)==0) + sum(trainLabel(trainPred>=t)==1)) / n_train;
    testAcc(i) = (sum(testLabel(testPred<t)==0) + sum(testLabel(testPred>=t)==1)) / n_test;
end
figure();
plot(thr, trainAcc, '.-', thr, testAcc, '^-');
ylim([0.45 1]); grid on;
h1 = legend('Train','Test');
h2 = xlabel('Threshold'); h3 = ylabel('Accuracy');
fprintf('Loss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\n_test', ...
        loss_train, loss, 100*max(trainAcc), 100*max(testAcc));
    
figure();
n_size = sqrt(size(center, 2));
for i = 1:n_center
    subplot(1, n_center, i);
    imshow(reshape(center(i, :), [n_size n_size]), [min(center(i, :)) max(center(i, :))]);
    title(sprintf('Cluster: %d', i), 'Interpreter', 'latex');
end

figure();
mean0 = mean(trainData(trainLabel==0, :), 1);
mean1 = mean(trainData(trainLabel==1, :), 1);
subplot(1, 2, 1);
imshow(reshape(mean0, [n_size n_size]), [min(mean0) max(mean0)]);
h4 = title('Mean: 1');
subplot(1, 2, 2);
imshow(reshape(mean1, [n_size n_size]), [min(mean1) max(mean1)]);
h5 = title('Mean: 2');

if n_center == 2
    figure();
	if sum(abs(mean0 - center(1, :))) < sum(abs(mean1 - center(1, :)))
        residual0 = abs(mean0 - center(1, :));
        residual1 = abs(mean1 - center(2, :));
    else
        residual0 = abs(mean1 - center(1, :));
        residual1 = abs(mean0 - center(2, :));
    end
    subplot(1, 2, 1);
    imshow(reshape(residual0, [n_size n_size]), [min(residual0) max(residual0)]);
    title('Residual: 1', 'Interpreter', 'latex');
    subplot(1, 2, 2);
    imshow(reshape(residual1, [n_size n_size]), [min(residual0) max(residual1)]);
    title('Residual: 2', 'Interpreter', 'latex');
end

set([h1 h2 h3 h4 h5], 'Interpreter', 'latex');
