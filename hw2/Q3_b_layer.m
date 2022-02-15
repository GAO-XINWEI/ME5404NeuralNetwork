clc;
clear all;
close all;
%% Data input and format
dataset_0_dir = dir('group_0/airplane/*.jpg');
dataset_1_dir = dir('group_0/cat/*.jpg');
n_train = 450;
n_test = 50;
img_train = zeros(1024, n_train*2);
img_test = zeros(1024, n_test*2);
label_train = zeros(1, n_train*2);
label_test = zeros(1, n_test*2);
for i = 1:n_train
    img_name = dataset_0_dir(i).name;
    img = imread(['group_0/airplane/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_train(:, i) = img_v(1:1024);
    label_train(i) = 0;
    
    img_name = dataset_1_dir(i).name;
    img = imread(['group_0/cat/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_train(:, i+n_train) = img_v(1:1024);
    label_train(i+n_train) = 1;
end
for i = 1:n_test
    img_name = dataset_0_dir(i+n_train).name;
    img = imread(['group_0/airplane/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_test(:, i) = img_v(1:1024);
    label_test(i) = 0;
    
    img_name = dataset_1_dir(i+n_train).name;
    img = imread(['group_0/cat/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_test(:, i+n_test) = img_v(1:1024);
    label_test(i+n_test) = 1;
end
fprintf('Data done!\n');

%% b) Normalization (Layer Norm)
dataset = [img_train, img_test];
value_mean = mean(dataset);
value_var = std(dataset,1);%1:std(dataset,1).^2 %2:std(dataset,1)
dataset = dataset - value_mean;
dataset = dataset ./ value_var;
img_train = dataset(:, 1:n_train*2);
img_test = dataset(:, (n_train*2+1):(n_train*2+n_test*2));

%% a) Rosenblatt’s perceptron (single layer perceptron)
net = perceptron;
net.performFcn = 'mse';
net.trainParam.lr=0.0001;
net.trainParam.epochs = 1;
max_epoch = 2000;
accu_train = zeros(1, max_epoch);
accu_test = zeros(1, max_epoch);
for epoch = 1:max_epoch
    idx = randperm(n_train*2);
    net = train(net, img_train(:, idx), label_train(:, idx));
    label_pred0 = net(img_train);
    label_pred = net(img_test);
    accu_train(epoch) = 1 - mean(abs(label_pred0 - label_train));
    accu_test(epoch) = 1 - mean(abs(label_pred - label_test));
    if mod(epoch,100)==0
        disp(epoch);
    end
end
figure();
plot(accu_train, 'linewidth', 1.5);
hold on;
plot(accu_test, 'linewidth', 1.5);
xlabel('Epoch'), ylabel('Accuracy'), title('Single layer perceptron (Layer Norm)');
legend('Train Accuracy', 'Test Accuracy', 'location', 'southeast');
fprintf('Accuracy of single layer network (train): %.4f\n', accu_train(end));
fprintf('Accuracy of single layer network (test): %.4f\n', mean(accu_test(end-100:end)));
