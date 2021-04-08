function [error_AEE,error_ASE,t_AEE,t_ASE]=GraphEncoderEvaluate(X,Y,num)

if nargin<3
    num=10; % 10 fold cross validation by default
end
Y=Y+1-min(Y);
indices = crossvalind('Kfold',Y,num);d=30;

error_AEE=zeros(num,1);t_AEE=zeros(num,1);error_ASE=zeros(num,d);t_ASE=zeros(num,d);
for i = 1:num
    %     test = (indices == i); % test indices
    %     train = ~test; % training indices
    
    test = (indices == i); % test indices
    train = ~test; % training indices
    
    tic
    [Z,W]=GraphEncoder(X(train,train),Y(train));
    mdl=fitcdiscr(Z,Y(train),'discrimType','pseudoLinear');
    tt=predict(mdl,X(test,train)*W);
    t_AEE(i)=toc;
    error_AEE(i)=error_AEE(i)+mean(Y(test)~=tt);
    
    tic
    [U,S,~]=svds(X,d);
    t1=toc;
    for j=1:d
        tic
        Z=U(:,1:j)*S(1:j,1:j)^0.5;
        mdl=fitcdiscr(Z(train,:),Y(train));
        tt=predict(mdl,Z(test,:));
        t_ASE(i,j)=t1+toc;
        error_ASE(i,j)=error_ASE(i,j)+mean(Y(test)~=tt);
    end
end

error_AEE=mean(error_AEE);
t_AEE=mean(t_AEE);
[error_ASE,ind]=min(mean(error_ASE,1));
t_ASE=mean(t_ASE,1); t_ASE=t_ASE(ind);
