% Matric No. A0243230X
% Ignore class 0 and class 1
clear all;
close all;

iter = 1000;
init_rate = 0.1; 
vert_neur = 10;
hor_neur = 10;
M = vert_neur * hor_neur;
SOM = rand(784,M);
init_width = sqrt( (10^2 + 10^2)) / 2;
label(1:vert_neur,1:hor_neur) = inf;

load('digits.mat')
trainIdx = find(train_classlabel ~= 0 & train_classlabel ~= 1);
trainData = train_data(:,trainIdx);
trainLabel = train_classlabel(:,trainIdx);
testIdx = find(test_classlabel ~= 0 & test_classlabel ~= 1);
testData = test_data(:,testIdx);
testLabel = test_classlabel(:,testIdx);
n_train = length(trainLabel);

for n = 0:iter
    rate = init_rate * exp(-n/iter);
    T1 = iter/(log(init_width));
    width = init_width * exp(-n /T1);
    
    for idx = 1:600
        sample = trainData(:,idx);
        % getwinner
        for i = 1:M
            dis(1,i) = norm(SOM(:,i)-sample);
        end
        winner = find(dis==min(dis));
        winner = winner(1,1);
        grid_col = mod(winner,hor_neur);
        if grid_col == 0
            grid_col = 10;%n
        end
        grid_row = ceil(winner/vert_neur);%k
        % get neighbour
        for i = 1:vert_neur
            for j = 1:hor_neur
                d(i,j) = -1 * (norm( [i j] - [grid_row grid_col] ) )^2;%d
                h(i,j) = exp(d(i,j) / (2*width^2));%h
            end
        end
        % update
        label(grid_row,grid_col) = trainLabel(:,idx);
        reshape_h = reshape(h',[1,100]);
        for i = 1:M
            SOM(:,i) = SOM(:,i) + rate * reshape_h(1,i) * (sample - SOM(:,i));
        end
    end
end

reshaped_label = reshape(label',[1 100]);

for A = 1:100
    subplot(10,10,A)
    graph = reshape(2*(SOM(:,A)),[28 28]);
    imshow(graph');
    title(sprintf('%0d',reshaped_label(1,A)));
end

[train_acc, train_result] = getacc(trainData,trainLabel,SOM,reshaped_label);
[test_acc, test_result] = getacc(testData,testLabel,SOM,reshaped_label);

function [accuracy, test_grid] = getacc(data,data_label,SOM,som_labels)
num_inputs = size(data,2);
num_weights = size(SOM,2);
test_grid = zeros(1,num_inputs);
for i = 1:num_inputs
    min = inf;
    for j = 1:num_weights
        diff = norm(data(:,i) - SOM(:,j));
        if diff < min
            min = diff;
            min_idx = j;
        end
    end
    test_grid(1,i) = som_labels(1,min_idx);
end
counter = 0;
for i = 1:num_inputs
    if test_grid(1,i) == double(data_label(1,i))
        counter = counter + 1;
    end
end
accuracy = counter/num_inputs;
end
