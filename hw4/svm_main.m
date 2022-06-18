clc
clear
%% load data
load('train.mat');
load('test.mat');
load('eval.mat');

best_acc_eval = 0;
best_gamma = 0;
best_C = 0;
%% Preprocessing input data
mu=mean(train_data,2);
std=std(train_data,1,2);
norm_train=(train_data-mu)./std;
% norm_test=(test_data-mu)./std;
norm_eval=(eval_data-mu)./std;
train_dim=size(norm_train,2);
%% Mercer condition check
% calculate rbf kernel
% for gamma = [0.00001,0.0001,0.001,0.01,0.1]
for gamma = [0.0001,0.001,0.01]
fprintf('searching...\n')
kernel=zeros(train_dim,train_dim);
for i=1:train_dim
    for j=1:train_dim
        kernel(i,j)=exp(-gamma*sum((norm_train(:,i)-norm_train(:,j)).^2));
    end
end
% determine
eigenvalues=eig(kernel);
flag = true;
if min(eigenvalues)<-1e-4
    flag = false;
    fprintf('this kernel candidate is not admissible')
end
%% Training
%%hyparameters
% for C = [0.1,1,10,100,1000,10000,100000,1000000]
for C = [10,100,1000]
if flag == true
    % Set the training parameters
    A = [];
    b = [];
    Aeq = train_label';
    Beq = 0;
    [f_dim,train_dim]=size(norm_train);
    lb = zeros(train_dim,1);
    ub=ones(train_dim,1)*C;
    f=-ones(train_dim,1);
    x0 = [];
    %calculate kernel and H
    H=zeros(train_dim,train_dim);
    for i=1:train_dim
        for j=1:train_dim
            kernel=exp(-gamma*sum((norm_train(:,i)-norm_train(:,j)).^2));
            D=train_label(i)*train_label(j);
            H(i,j)=D.*kernel;
        end
    end
    options=optimset('MaxIter',1000);
    Alpha = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);
    % select support vectors
    idx = find(Alpha>1e-8);
    % Calculate disciminant parameters
    num_SVM=length(idx);    
    support_label=train_label(idx);
    bN=zeros(num_SVM,1);
    for i=1:num_SVM
        more_k=Alpha.*train_label.*exp(-gamma*sum((norm_train-norm_train(:,idx(i))).^2,1)');
        bN(i)=support_label(i)-sum(more_k);
    end
    bo=mean(bN);
    % performance evaluate
    [~,acc_train] = Acc(Alpha,bo,norm_train,train_label,norm_train,train_label,gamma);
    % [~,acc_test]= Acc(Alpha,bo,norm_train,train_label,norm_test,test_label,gamma);
    [eval_predicted,acc_eval]= Acc(Alpha,bo,norm_train,train_label,norm_eval,eval_label,gamma);
    % record the result
    if best_acc_eval < acc_eval
        best_acc_eval = acc_eval;
        best_gamma = gamma;
        best_C = C;
    end
end
end
end
fprintf('search done!\n')
fprintf('Our result is:%.2f%% when gimma=%f C=%.2f\n ',best_acc_eval*100,best_gamma,best_C)

%% functions
function [pred_label,accuracy]  = Acc(alpha,bo,train_data,train_label,data,label,gamma)
    g=zeros(length(label),1);
    for i=1:length(label)
        more_K=alpha.*train_label.*exp(-gamma*sum((train_data-data(:,i)).^2,1)');
        g(i)=sum(more_K)+bo;
    end
    pred_label=sign(g);
    accuracy = mean(pred_label == label,'all');
end
