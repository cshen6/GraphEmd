%%%%%%%%%%%%%%
n=2000;
[A,Y]=simGenerate(11,n,5);
% E=adj2edge(A);

[A,Y]=simGenerate(21,n,5,0);
% A=edge2adj(E);
% 
% %% Simulate Data
% K=3;n=1000;
% B=0.1*ones(K,K);B(1,1)=0.15;B(2,2)=0.13;
% Y=randi(K,n,1);
% A=zeros(n,n);
% for i=1:n
% for j=i+1:n
% A(i,j)=rand(1)<B(Y(i),Y(j));
% end
% end
% A=A+A';

%% Preparing evaluation
fold=5;
cvp = cvpartition(size(A, 1), 'KFold', fold);
errorGCN=zeros(fold,1);
errorGEE=zeros(fold,1);
errorGSE=zeros(fold,1);
time=zeros(1,3);

%% For GCN: Process into Normalized Graph Adjacency for GCN
tic
A2=A+eye(n);
D=sum(A2);
D=diag(D);
A3=D^-0.5*A2*D^-0.5;
time(1)=time(1)+toc;
%% For GSE: Graph Spectral embedding into d=3
tic
[U,S,V]=svds(sparse(A));d=3;
Z2=U(:,1:d)*S(1:d,1:d)^0.5;
time(2)=time(2)+toc;

for fold = 1:fold
    trainIndices = training(cvp, fold);
    testIndices = test(cvp, fold);
    
    % evaluate GCN
    tic
    trainData = A3(trainIndices, :);
    testData = A3(testIndices, :);
    mdlGCN=fitcnet(trainData,Y(trainIndices),'LayerSize',10);
    ytest=predict(mdlGCN,testData);
    errorGCN(fold)=mean(Y(testIndices)~=ytest);
    time(1)=time(1)+toc;

    % evaluate GSE
    tic
    mdlGSE=fitcdiscr(Z2(trainIndices,:),Y(trainIndices));
    ytest=predict(mdlGSE,Z2(testIndices,:));
    errorGSE(fold)=mean(Y(testIndices)~=ytest);
    time(2)=time(2)+toc;

    % evaluate GEE
    tic
    Y2=Y;
    Y2(testIndices)=0;
    Z=GraphEncoder(A,Y2);
    mdlGEE=fitcdiscr(Z(trainIndices,:),Y(trainIndices));
    ytest=predict(mdlGEE,Z(testIndices,:));
    errorGEE(fold)=mean(Y(testIndices)~=ytest); 
    time(3)=time(3)+toc;
end
[mean(errorGCN),mean(errorGSE),mean(errorGEE)]
[std(errorGCN),std(errorGSE),std(errorGEE)]
time