function [error_AEE,error_AEN,error_ARE,error_ARN,error_ASE,t_AEE,t_AEN,t_ARE,t_ARN,t_ASE]=GraphEncoderEvaluate2(X,Y,opts)

if nargin < 3
    opts = struct('kfold',10,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','purelin','normalize',0); % default parameters
end
if ~isfield(opts,'kfold'); opts.kfold=10; end
if ~isfield(opts,'knn'); opts.knn=5; end
if ~isfield(opts,'dim'); opts.dim=30; end
if ~isfield(opts,'neuron'); opts.neuron=10; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'training'); opts.training=0.8; end
if ~isfield(opts,'activation'); opts.activation='purelin'; end
if ~isfield(opts,'normalize'); opts.normalize=0; end
warning('off','all');
d=opts.dim; num=opts.kfold;

[~,~,Y]=unique(Y);

Y2=zeros(length(Y),length(unique(Y)));
for i=1:length(Y)
    Y2(i,Y(i))=1;
end
indices = crossvalind('Kfold',Y,num);
if opts.normalize==1
    deg=diag(sum(X));
    %X=deg^-1*X;
    X=deg^-0.5*X*deg^-0.5;
end

%% change adjacency to edge
Adj=X;
% if size(X,2)==size(X,1)
%     Edge=zeros(size(X,1)^2,1);s=0;
%     for i=1:size(X,1)
%         for j=i+1:size(X,1)
%             if X(i,j)>0
%                 s=s+1;
%                 Edge(s,1)=i;
%                 Edge(s,2)=j;
%             end
%         end
%     end
%     X=Edge(1:s,:);
% end

% initialize NN
net = patternnet(opts.neuron,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
net.layers{1}.transferFcn = opts.activation;
net.trainParam.showWindow = false;
net.trainParam.epochs=opts.epoch;
net.divideParam.trainRatio = opts.training;
net.divideParam.valRatio   = 1-opts.training;
net.divideParam.testRatio  = 0/100;
    
error_AEE=zeros(num,1);error_ARE=zeros(num,1);error_ARN=zeros(num,1);error_AEN=zeros(num,1);t_AEE=zeros(num,1);t_ARE=zeros(num,1);t_ARN=zeros(num,1);t_AEN=zeros(num,1);error_ASE=zeros(num,d);t_ASE=zeros(num,d);
rep=100;
for i = 1:opts.kfold
    %     tst = (indices == i); % tst indices
    %     trn = ~tst; % trning indices
    
    tst = (indices == i); % tst indices
    trn = ~tst; % trning indices
    
    tic
    [Z,W]=GraphEncoder(X(trn,trn),Y(trn));
    tmp=toc;
    
    tic
    mdl=fitcdiscr(Z,Y(trn),'discrimType','pseudoLinear');
    tt=predict(mdl,X(tst,trn)*W);
    t_AEE(i)=tmp+toc;
    error_AEE(i)=error_AEE(i)+mean(Y(tst)~=tt);
    
    tic
    Y2Trn=Y2(trn,:);  
    mdl3 = train(net,Z',Y2Trn');
    classes = mdl3((X(tst,trn)*W)'); % class-wise probability for tsting data
    %error_NN = perform(mdl3,Y2Tsn',classes);
    tt = vec2ind(classes)'; % this gives the actual class for each observation
    t_AEN(i)=tmp+toc;
    error_AEN(i)=error_AEN(i)+mean(Y(tst)~=tt);
    
    tic
    [Z,W]=GraphEncoders(X(trn,trn),Y(trn),rep);
    tmp=toc;
    
    tic
    tt=zeros(rep,sum(tst));
    for r=1:rep
        mdl=fitcdiscr(Z(:,:,r),Y(trn),'discrimType','pseudoLinear');
        tt(r,:)=predict(mdl,X(tst,trn)*W(:,:,r));
    end
    t_ARE(i)=tmp+toc;
    error_ARE(i)=error_ARE(i)+mean(Y(tst)~=mode(tt,1)');
%     tic
%     tt=zeros(rep,sum(tst));
%     for r=1:rep
%         mdl=fitcknn(Z(:,:,r),Y(trn),'Distance','cosine','NumNeighbors',opts.knn);
%          tt(r,:)=predict(mdl,X(tst,trn)*W);
%     end
%     t_ARE(i)=tmp+toc;
%     error_ARE(i)=error_ARE(i)+mean(Y(tst)~=mode(tt,1)');
    
    tic
    Y2Trn=Y2(trn,:);
    tt=zeros(rep,sum(tst));
    for r=1:rep
        mdl3 = train(net,Z(:,:,r)',Y2Trn');
        classes = mdl3((X(tst,trn)*W(:,:,r))'); % class-wise probability for tsting data
        %error_NN = perform(mdl3,Y2Tsn',classes);
         tt(r,:) = vec2ind(classes)'; % this gives the actual class for each observation
    end  
    t_ARN(i)=tmp+toc;
    error_ARN(i)=error_ARN(i)+mean(Y(tst)~=mode(tt,1)');
    
    tic
    [U,S,~]=svds(Adj,d);
    t1=toc;
    for j=1:d
        tic
        Z=U(:,1:j)*S(1:j,1:j)^0.5;
        mdl=fitcdiscr(Z(trn,:),Y(trn));
        tt=predict(mdl,Z(tst,:));
        t_ASE(i,j)=t1+toc;
        error_ASE(i,j)=error_ASE(i,j)+mean(Y(tst)~=tt);
    end
end

error_AEE=mean(error_AEE);
t_AEE=mean(t_AEE);
error_AEN=mean(error_AEN);
t_AEN=mean(t_AEN);
error_ARN=mean(error_ARN);
t_ARN=mean(t_ARN);
error_ARE=mean(error_ARE);
t_ARE=mean(t_ARE);
[error_ASE,ind]=min(mean(error_ASE,1));
t_ASE=mean(t_ASE,1); t_ASE=t_ASE(ind);