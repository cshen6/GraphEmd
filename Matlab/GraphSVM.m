function [error]=GraphSVM(X,Y,option,d,num)

if nargin<3
   option=0; % 0 for SVM, 1 for MDS * LDA, 2 for ASE * LDA, 3 for RF
end
if nargin<4
    d=3; % dimension
end
if nargin<5
    num=10; % 10 fold cross validation
end
%opt='euclidean';
opt='hsic';
if option<4
    [~,ind]=checkDist(X);
    if ind==0
        X=DCorInput(X,opt);
    end
end

% Adjacency Spectral Embedding
if option==1
    n=size(X,1);
    H=eye(n)-ones(n)/n;
    [U,S,~]=svd(-H*X*H/2);
    X=U(:,1:d)*S(1:d,1:d)^(0.5);
end
if option==2
    [U,S,~]=svd(X);
    X=U(:,1:d)*S(1:d,1:d)^(0.5);
end

% Cross Validation using Linear Dsicriminant
indices = crossvalind('Kfold',Y,num);
error=0;
% rng('default')
for i = 1:num
    test = (indices == i); 
    train = ~test;
    switch option
        case 0
            mdl=fitcecoc(X(train,:),Y(train));
        case 1
            mdl=fitcdiscr(X(train,:),Y(train),'discrimType','pseudoLinear'); 
        case 2
            mdl=fitcdiscr(X(train,:),Y(train)); 
        case 3
            mdl=TreeBagger(100,X(train,:),Y(train),'Method','classification');
    end
    if option==3
        tt=str2double(predict(mdl,X(test,:)));
    else
        tt=predict(mdl,X(test,:));
    end
    error=error+mean(Y(test)~=tt)/num;
end