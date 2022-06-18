% Set the training parameters
% Hard margin: +inf(in theory)/10e6 (in practice)
A = [];
b = [];
Aeq = train_label';
Beq = 0;
[f_dim,s_dim]=size(norm_train);
lb = zeros(s_dim,1);
C = 10e6; 
ub=ones(s_dim,1)*C;
f=-ones(s_dim,1);
x0 = [];
H_sign = train_label*train_label';
H = norm_train'*norm_train.*H_sign;
options = optimset('MaxIter',200);
Alpha = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);
% select support vectors
idx = find(Alpha>1e-4);
% Calculate disciminant parameters
wo = sum(Alpha'.*train_label'.*norm_train,2);
bo=mean(1./train_label(idx) - norm_train(:,idx)'*wo);
