% Matric No. A0243230X
% Ignore class 0 and class 1

% Note: plase run after 'q3_c1.m'!
close all;
counter_1 = 1;
counter_2 = 1;
for i = 1:60
    if counter_1 <= 4 || counter_2 <= 4
        [test_acc, test_result_, min_idx] = getacc_(testData(:,i),testLabel(:,i),SOM,reshaped_label);
        if test_acc == 1 && counter_1 <= 4
            figure(1)
            sgtitle('Correct classification')
            subplot(4,2,(counter_1-1)*2+1)
            imshow(reshape(testData(:,i),28,28))
            title(sprintf('Truth:%d',testLabel(1,i)))
            subplot(4,2,(counter_1-1)*2+2)
            imshow(reshape(SOM(:,min_idx),28,28))
            title(sprintf('Predicted:%d',test_result(1,i)))
            counter_1 = counter_1 + 1;
        end
        if test_acc == 0 && counter_2 <= 4
            figure(2)
            sgtitle('Incorrect classification')
            subplot(4,2,(counter_2-1)*2+1)
            imshow(reshape(testData(:,i),28,28))
            title(sprintf('Truth:%d',testLabel(1,i)))
            subplot(4,2,(counter_2-1)*2+2)
            imshow(reshape(SOM(:,min_idx),28,28))
            title(sprintf('Predicted:%d',test_result(1,i)))
            counter_2 = counter_2 + 1;
        end
    end
end


function [accuracy, test_grid, min_idx] = getacc_(data,data_label,SOM,som_labels)
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
if counter == 0
    accuracy = 0;
else
    accuracy = counter/num_inputs;
end
end
    