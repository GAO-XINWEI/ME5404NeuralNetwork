for i=1:train_dim
    for j=1:train_dim
        kernel=exp(-gamma*sum((norm_train(:,i)-norm_train(:,j)).^2));
        D=train_label(i)*train_label(j);
        H(i,j)=D.*kernel;
    end
end