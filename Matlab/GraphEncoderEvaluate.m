function [error_AEE,error_AEE2,error_AEN,error_ASE,t_AEE,t_AEE2,t_AEN,t_ASE]=GraphEncoderEvaluate(X,Y,opts)

if nargin < 3
    opts = struct('kfold',10,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','purelin'); % default parameters
end
if ~isfield(opts,'kfold'); opts.kfold=10; end
if ~isfield(opts,'knn'); opts.knn=5; end
if ~isfield(opts,'dim'); opts.dim=30; end
if ~isfield(opts,'neuron'); opts.neuron=10; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'training'); opts.training=0.8; end
if ~isfield(opts,'activation'); opts.activation='purelin'; end
warning('off','all');
d=opts.dim; 

[~,~,Y]=unique(Y);
n=length(Y);
k=max(Y);
Y2=zeros(n,k);
for i=1:n
    Y2(i,Y(i))=1;
end
indices = crossvalind('Kfold',Y,opts.kfold);
num=size(X,3);

% initialize NN
net = patternnet(opts.neuron,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
net.layers{1}.transferFcn = opts.activation;
net.trainParam.showWindow = false;
net.trainParam.epochs=opts.epoch;
net.divideParam.trainRatio = opts.training;
net.divideParam.valRatio   = 1-opts.training;
net.divideParam.testRatio  = 0/100;

error_AEE=zeros(opts.kfold,1);error_AEE2=zeros(opts.kfold,1);error_AEN=zeros(opts.kfold,1);t_AEE=zeros(opts.kfold,1);t_AEE2=zeros(opts.kfold,1);t_AEN=zeros(opts.kfold,1);error_ASE=zeros(opts.kfold,d);t_ASE=zeros(opts.kfold,d);
for i = 1:opts.kfold
    %     tst = (indices == i); % tst indices
    %     trn = ~tst; % trning indices
    
    tsn = (indices == i); % tst indices
    trn = ~tsn; % trning indices
    
    tic
    [Z,W]=GraphEncoder(X(trn,trn,:),Y(trn));
    tmp=toc;
    
    tic
    mdl=fitcdiscr(Z,Y(trn),'discrimType','pseudoLinear');
    if num>1
        XTsn=zeros(sum(tsn),k,num);
        for r=1:num
            XTsn(:,:,r)=X(tsn,trn,r)*W;
        end
        XTsn=reshape(XTsn,sum(tsn),k*num);
    else
        XTsn=X(tsn,trn)*W;
    end
    tt=predict(mdl,XTsn);
    t_AEE(i)=tmp+toc;
    error_AEE(i)=error_AEE(i)+mean(Y(tsn)~=tt);
    
    tic
    mdl=fitcknn(Z,Y(trn),'Distance','cosine','NumNeighbors',opts.knn);
    tt=predict(mdl,XTsn);
    t_AEE2(i)=tmp+toc;
    error_AEE2(i)=error_AEE2(i)+mean(Y(tsn)~=tt);
    
    tic
%     rep=100;
%     [Z,W]=GraphEncoders(X(trn,trn),Y(trn),rep);
%     Z=reshape(Z,sum(trn),k*rep);
%     W=reshape(W,sum(trn),k*rep);
    Y2Trn=Y2(trn,:);  
    mdl3 = train(net,Z',Y2Trn');
    classes = mdl3(XTsn'); % class-wise probability for tsting data
    %error_NN = perform(mdl3,Y2Tsn',classes);
    tt = vec2ind(classes)'; % this gives the actual class for each observation
    t_AEN(i)=tmp+toc;
    error_AEN(i)=error_AEN(i)+mean(Y(tsn)~=tt);
    
    if num==1
    tic
    [U,S,~]=svds(X,d);
    t1=toc;
    for j=1:d
        tic
        Z=U(:,1:j)*S(1:j,1:j)^0.5;
        mdl=fitcdiscr(Z(trn,:),Y(trn));
        tt=predict(mdl,Z(tsn,:));
        t_ASE(i,j)=t1+toc;
        error_ASE(i,j)=error_ASE(i,j)+mean(Y(tsn)~=tt);
    end
    end
end

error_AEE=mean(error_AEE);
t_AEE=mean(t_AEE);
error_AEN=mean(error_AEN);
t_AEN=mean(t_AEN);
error_AEE2=mean(error_AEE2);
t_AEE2=mean(t_AEE2);
[error_ASE,ind]=min(mean(error_ASE,1));
t_ASE=mean(t_ASE,1); t_ASE=t_ASE(ind);
