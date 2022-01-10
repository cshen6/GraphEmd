function [result]=GraphEncoderEvaluate(X,Y,opts)

if nargin < 3
    opts = struct('indices',crossvalind('Kfold',Y,5),'Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',20,'epoch',100,'training',0.05,'activation','poslin','Learner',1); % default parameters
end
if ~isfield(opts,'indices'); opts.indices=crossvalind('Kfold',Y,5); end
if ~isfield(opts,'Adjacency'); opts.Adjacency=1; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=1; end
if ~isfield(opts,'Spectral'); opts.Spectral=1; end
if ~isfield(opts,'LDA'); opts.LDA=1; end
if ~isfield(opts,'GFN'); opts.GFN=0; end
if ~isfield(opts,'GCN'); opts.GCN=0; end
if ~isfield(opts,'GNN'); opts.GNN=0; end
if ~isfield(opts,'Learner'); opts.Learner=1; end
if ~isfield(opts,'LearnIter'); opts.LearnIter=0; end
if ~isfield(opts,'knn'); opts.knn=5; end
if ~isfield(opts,'dim'); opts.dim=30; end
% if ~isfield(opts,'deg'); opts.deg=0; end
if ~isfield(opts,'neuron'); opts.neuron=20; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'training'); opts.training=0.05; end
if ~isfield(opts,'activation'); opts.activation='poslin'; end %purelin, tansig
warning('off','all');
%met=[opts.AEE,opts.LDA,opts.GFN,opts.ASE,opts.LSE,opts.GCN,opts.GNN]; %AEE, LDA, GFN, ASE, GFN, ANN
indices=opts.indices;
if length(indices)~=length(Y)
    indices=crossvalind('Kfold',Y,5);
end

kfold=max(indices);
[~,~,Y]=unique(Y);
n=length(Y);
d=min(opts.dim,n-1);
K=max(Y);
num=size(X,3);
% nre=opts.resample;
ide=eye(n);
klim=20;
opts.knn=min(opts.knn,ceil(n/K/3));
discrimType='pseudoLinear';
% if k>10
%     discrimType='diagLinear';
% end
%%Edge to Adj
% if size(X,2)==2
%     Adj=zeros(n,n);
%     for i=1:size(X,1)
%         Adj(X(i,1),X(i,2))=1;
%     end
%     X=Adj+Adj';
% end
% [X2]=adj2edge(X);

% initialize NN
netGNN = patternnet(max(opts.neuron,K),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
netGNN.layers{1}.transferFcn = opts.activation;
netGNN.trainParam.showWindow = false;
netGNN.trainParam.epochs=opts.epoch;
netGNN.divideParam.trainRatio = 0.9;
netGNN.divideParam.valRatio   = 0.1;
netGNN.divideParam.testRatio  = 0/100;

netGFN = patternnet(max(opts.neuron,K),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
netGFN.layers{1}.transferFcn = opts.activation;
netGFN.trainParam.showWindow = false;
netGFN.trainParam.epochs=opts.epoch;
netGFN.divideParam.trainRatio = 0.9;
netGFN.divideParam.valRatio   = 0.1;
netGFN.divideParam.testRatio  = 0/100;
%netGFN.trainParam.lr = 0.01;

% netGFN = patterNNt(max(opts.neuron,k),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
% netGFN.layers{1}.transferFcn = opts.activation;
% netGFN.trainParam.showWindow = false;
% netGFN.trainParam.epochs=opts.epoch;
% netGFN.divideParam.trainRatio = opts.training;
% netGFN.divideParam.valRatio   = 1-opts.training;
% netGFN.divideParam.testRatio  = 0/100;
%patterNNt(10,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
%net1.numLayers = 1;
%net1.layers{1}=softmaxLayer;

%% GCN paramers
num_epoch = 100;        % Number of epochs
d2 = 10;               % Number of hidden units
learning_rate = 1e-4;  % The alpha parameter in the ADAM optimizer
l2_reg = 0;            % L2 regularization weight
batch_size = [];      % Batch size. If empty, equivalent to GCN w/o batching
sample_size = [];     % Sample size. If empty, equivalent to batched GCN
szW0 = [n,d2];       % Size of parameter matrix W0
szW1 = [d2,K];       % Size of parameter matrix W1
num_var = prod(szW0) + prod(szW1);
% adam_param = adam_init(num_var, learning_rate);

acc_AEE_NN=zeros(kfold,1);acc_AEE_LDA=zeros(kfold,1);acc_GFN=zeros(kfold,1);acc_GNN=zeros(kfold,1);
acc_LEE_NN=zeros(kfold,1);acc_LEE_LDA=zeros(kfold,1);t_LEE_NN=zeros(kfold,1);t_LEE_LDA=zeros(kfold,1);
t_AEE_NN=zeros(kfold,1);t_AEE_LDA=zeros(kfold,1);t_GFN=zeros(kfold,1);t_GNN=zeros(kfold,1);
acc_ASE_NN=zeros(kfold,d);t_ASE_NN=zeros(kfold,d);acc_ASE_LDA=zeros(kfold,d);t_ASE_LDA=zeros(kfold,d);acc_LSE_NN=zeros(kfold,d);t_LSE_NN=zeros(kfold,d);acc_LSE_LDA=zeros(kfold,d);t_LSE_LDA=zeros(kfold,d);
acc_GCN=zeros(kfold,1);t_GCN=zeros(kfold,1);
% opts1 = struct('deg',opts.deg); % default parameters
% opts2 = struct('deg',opts.deg,'pivot',opts.pivot); % default parameters

for i = 1:kfold
    if opts.LearnIter==0
        %     tst = (indices == i); % tst indices
        %     trn = ~tst; % trning indices
        
        tsn = (indices == i); % tst indices
        trn = ~tsn; % trning indices
        
        %     trn = (indices == i); % tst indices
        %     tsn = ~trn; % trning indices
        
        val = (indices == max(mod(i+1,kfold+1),1));
        %     trn2= ~(tsn+val);
        
        if opts.Adjacency==1
            YT=Y;
            YT(tsn)=-1;
            YTrn=Y(trn);
            YTsn=Y(tsn);
            tic
            oot=struct('Laplacian',false,'LearnIter',0,'Learner',opts.Learner);
            [Z,YTNew,~,indT]=GraphEncoder(X,YT,oot);
            ZTrn=Z(indT,:);
            ZTsn=Z(~indT,:);
            
            %     else
            %         [Z,indT]=GraphEncoder(X,YT,opts);
            %         %[Z,indT]=GraphSBMEst(X,YT);
            %     end
            %     if k>klim
            %         [ind,~,~] = DCorScreening(Z,Y(trn));
            %         Z=Z(:,ind);
            %     end
            tmp1=toc;
            
            if opts.knn>0
                tic
                mdl=fitcknn(ZTrn,YTrn,'Distance','euclidean','NumNeighbors',opts.knn);
                tt=predict(mdl,ZTsn);
                t_AEE_NN(i)=tmp1+toc;
                acc_AEE_NN(i)=acc_AEE_NN(i)+mean(Y(tsn)~=tt);
            end
            
            if opts.LDA==1
                tic
%                 mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
%                 tt=predict(mdl,ZTsn);
                t_AEE_LDA(i)=tmp1+toc;
                acc_AEE_LDA(i)=acc_AEE_LDA(i)+mean(YTNew(tsn)~=tt);
            end
            
            
            if opts.GFN==1
                tic
                Y2=zeros(length(YTrn),K);
                for j=1:length(YTrn)
                    Y2(j,YTrn(j))=1;
                end
                %         Y2Trn=Y2(trn,:);
                mdl3 = train(netGFN,ZTrn',Y2');
                classes = mdl3(ZTsn'); % class-wise probability for tsting data
                %acc_NN = perform(mdl3,Y2Tsn',classes);
                tt = vec2ind(classes)'; % this gives the actual class for each observation
                t_GFN(i)=tmp1+toc;
                acc_GFN(i)=acc_GFN(i)+mean(YTsn~=tt);
            end
            
            if opts.Spectral==1
                % ASE
                Adj=mean(X,3);YA=Y;
                tic
                %         if nre>n
                %             Adj=AdjRe;
                %             trn=1:nre;
                %             tsn=nre+1:size(AdjRe,1);
                %             YA=YRe;
                %         end
                [U,S,~]=svds(Adj,d);
                t1=toc;
                for j=1:d
                    tic
                    Z=U(:,1:j)*S(1:j,1:j)^0.5;
                    t2=toc;
                    if opts.LDA==1
                        tic
                        mdl=fitcdiscr(Z(trn,:),YA(trn),'discrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_LDA(i,j)=t1;%+t2+toc;
                        acc_ASE_LDA(i,j)=acc_ASE_LDA(i,j)+mean(YTsn~=tt);
                    end
                    if opts.knn>0
                        tic
                        mdl=fitcknn(Z(trn,:),YA(trn),'Distance','euclidean','NumNeighbors',opts.knn);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_NN(i,j)=t1+t2+toc;
                        acc_ASE_NN(i,j)=acc_ASE_NN(i,j)+mean(YTsn~=tt);
                    end
                end
            end
        end
        
        if opts.Laplacian==1
            YT=Y;
            YT(tsn)=-1;
            oot=struct('Laplacian',true,'LearnIter',0,'Learner',opts.Learner);
            YTrn=Y(trn);
            YTsn=Y(tsn);
            tic
            [Z,~,~,indT]=GraphEncoder(X,YT,oot);
            ZTrn=Z(indT,:);
            ZTsn=Z(~indT,:);
            %     else
            %         [Z,indT]=GraphEncoder(X,YT,opts);
            %         %[Z,indT]=GraphSBMEst(X,YT);
            %     end
            %     if k>klim
            %         [ind,~,~] = DCorScreening(Z,Y(trn));
            %         Z=Z(:,ind);
            %     end
            tmp1=toc;
            if opts.knn>0
                tic
                mdl=fitcknn(ZTrn,YTrn,'Distance','euclidean','NumNeighbors',opts.knn);
                tt=predict(mdl,ZTsn);
                t_LEE_NN(i)=tmp1+toc;
                acc_LEE_NN(i)=acc_LEE_NN(i)+mean(Y(tsn)~=tt);
            end
            
            if opts.LDA==1
                tic
                mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
                tt=predict(mdl,ZTsn);
                t_LEE_LDA(i)=tmp1+toc;
                acc_LEE_LDA(i)=acc_LEE_LDA(i)+mean(YTsn~=tt);
            end
            
            
            % ASE
            if opts.Spectral==1
                tic
                Adj=mean(X,3);
                D=max(sum(Adj,1),1).^(0.5);
                AdjT=Adj;
                for j=1:n
                    AdjT(:,j)=AdjT(:,j)/D(j)./D';
                end
                [U,S,~]=svds(AdjT,d);
                t1=toc;
                for j=1:d
                    tic
                    Z=U(:,1:j)*S(1:j,1:j)^0.5;
                    t2=toc;
                    if opts.LDA==1
                        tic
                        mdl=fitcdiscr(Z(trn,:),Y(trn),'discrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_LSE_LDA(i,j)=t1;%+t2+toc;
                        acc_LSE_LDA(i,j)=acc_LSE_LDA(i,j)+mean(Y(tsn)~=tt);
                    end
                    if opts.knn>0
                        tic
                        mdl=fitcknn(Z(trn,:),Y(trn),'Distance','euclidean','NumNeighbors',opts.knn);
                        tt=predict(mdl,Z(tsn,:));
                        t_LSE_NN(i,j)=t1+t2+toc;
                        acc_LSE_NN(i,j)=acc_LSE_NN(i,j)+mean(Y(tsn)~=tt);
                    end
                end
            end
        end
        
        % kipf GCN
        
        %     if opts.GCN==1
        %         tic
        %         acc_GCN(i)=model_fastgcn_train_and_test(X, ide, Y2, trn2, val, tsn, ...
        %             szW0, szW1, l2_reg, num_epoch, batch_size, ...
        %             sample_size, adam_param);
        %         t_GCN(i)=toc;
        %     end
        
        %  Direct NN
        if opts.GNN==1
            tic
            X1=reshape(X(trn,trn,:),sum(trn),num*sum(trn));
            X2=reshape(X(tsn,trn,:),sum(tsn),num*sum(trn));
            mdl3 = train(netGNN,X1',Y2Trn');
            classes = mdl3(X2'); % class-wise probability for tsting data
            tt = vec2ind(classes)'; % this gives the actual class for each observation
            t_GNN(i)=tmp2+toc;
            acc_GNN(i)=acc_GNN(i)+mean(Y(tsn)~=tt);
        end
    else
        trn=(indices==0);
        for j=1:K
            tmp=find(Y==j);
            sz=ceil(length(tmp)*opts.training)+1;
            sz=randi(length(tmp),sz,1);
            trn(sz)=1;
        end
        tsn = ~trn; % trning indices
        YT=Y;
        YT(tsn)=-1;
        YTsn=Y(tsn);
        if opts.Adjacency==1
            oot=struct('LearnIter',opts.LearnIter,'Learner',opts.Learner);
            tic
            [~,tt,~,~]=GraphEncoder(X,YT,oot);
            t_AEE_LDA(i)=toc;
            acc_AEE_LDA(i)=mean(YTsn~=tt(tsn));
            if opts.Spectral==1
                % ASE
                Adj=mean(X,3);YA=Y;
                tic
                %         if nre>n
                %             Adj=AdjRe;
                %             trn=1:nre;
                %             tsn=nre+1:size(AdjRe,1);
                %             YA=YRe;
                %         end
                [U,S,~]=svds(Adj,d);
                t1=toc;
                for j=1:d
                    tic
                    Z=U(:,1:j)*S(1:j,1:j)^0.5;
                    t2=toc;
%                     if opts.LDA==1
                        tic
                        mdl=fitcdiscr(Z(trn,:),YA(trn),'discrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_LDA(i,j)=t1+t2+toc;
                        acc_ASE_LDA(i,j)=acc_ASE_LDA(i,j)+mean(YTsn~=tt);
%                     end
%                     if opts.knn>0
%                         tic
%                         mdl=fitcknn(Z(trn,:),YA(trn),'Distance','euclidean','NumNeighbors',opts.knn);
%                         tt=predict(mdl,Z(tsn,:));
%                         t_ASE_NN(i,j)=t1+t2+toc;
%                         acc_ASE_NN(i,j)=acc_ASE_NN(i,j)+mean(YTsn~=tt);
%                     end
                end
            end
        end
        if opts.Laplacian==1
            oot=struct('Laplacian',true,'LearnIter',opts.LearnIter,'Learner',opts.Learner);
            tic
            [~,tt]=GraphEncoder(X,YT,oot);
            t_LEE_LDA(i)=toc;
            acc_LEE_LDA(i)=mean(YTsn~=tt(tsn));
            if opts.Spectral==1
                tic
                Adj=mean(X,3);
                D=max(sum(Adj,1),1).^(0.5);
                AdjT=Adj;
                for j=1:n
                    AdjT(:,j)=AdjT(:,j)/D(j)./D';
                end
                [U,S,~]=svds(AdjT,d);
                t1=toc;
                for j=1:d
                    tic
                    Z=U(:,1:j)*S(1:j,1:j)^0.5;
                    t2=toc;
%                     if opts.LDA==1
                        tic
                        mdl=fitcdiscr(Z(trn,:),Y(trn),'discrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_LSE_LDA(i,j)=t1+t2+toc;
                        acc_LSE_LDA(i,j)=acc_LSE_LDA(i,j)+mean(Y(tsn)~=tt);
%                     end
%                     if opts.knn>0
%                         tic
%                         mdl=fitcknn(Z(trn,:),Y(trn),'Distance','euclidean','NumNeighbors',opts.knn);
%                         tt=predict(mdl,Z(tsn,:));
%                         t_LSE_NN(i,j)=t1+t2+toc;
%                         acc_LSE_NN(i,j)=acc_LSE_NN(i,j)+mean(Y(tsn)~=tt);
%                     end
                end
            end
        end
    end
end

std_AEE_NN=std(acc_AEE_NN);
acc_AEE_NN=1-mean(acc_AEE_NN);
t_AEE_NN=mean(t_AEE_NN);
std_AEE_LDA=std(acc_AEE_LDA);
acc_AEE_LDA=1-mean(acc_AEE_LDA);
t_AEE_LDA=mean(t_AEE_LDA);
std_LEE_NN=std(acc_LEE_NN);
acc_LEE_NN=1-mean(acc_LEE_NN);
t_LEE_NN=mean(t_LEE_NN);
std_LEE_LDA=std(acc_LEE_LDA);
acc_LEE_LDA=1-mean(acc_LEE_LDA);
t_LEE_LDA=mean(t_LEE_LDA);
std_GFN=std(acc_GFN);
acc_GFN=1-mean(acc_GFN);
t_GFN=mean(t_GFN);
std_ASE_NN=std(acc_ASE_NN);
[acc_ASE_NN,ind]=min(mean(acc_ASE_NN,1));
acc_ASE_NN=1-acc_ASE_NN;
std_ASE_NN=std_ASE_NN(ind);
t_ASE_NN=mean(t_ASE_NN,1); t_ASE_NN=t_ASE_NN(ind);
std_ASE_LDA=std(acc_ASE_LDA);
[acc_ASE_LDA,ind]=min(mean(acc_ASE_LDA,1));
acc_ASE_LDA=1-acc_ASE_LDA;
std_ASE_LDA=std_ASE_LDA(ind);
t_ASE_LDA=mean(t_ASE_LDA,1); t_ASE_LDA=t_ASE_LDA(ind);
std_LSE_NN=std(acc_LSE_NN);
[acc_LSE_NN,ind]=min(mean(acc_LSE_NN,1));
acc_LSE_NN=1-acc_LSE_NN;
std_LSE_NN=std_LSE_NN(ind);
t_LSE_NN=mean(t_LSE_NN,1); t_LSE_NN=t_LSE_NN(ind);
std_LSE_LDA=std(acc_LSE_LDA);
[acc_LSE_LDA,ind]=min(mean(acc_LSE_LDA,1));
acc_LSE_LDA=1-acc_LSE_LDA;
std_LSE_LDA=std_LSE_LDA(ind);
t_LSE_LDA=mean(t_LSE_LDA,1); t_LSE_LDA=t_LSE_LDA(ind);
std_GNN=std(acc_GNN);
acc_GNN=1-mean(acc_GNN);
t_GNN=mean(t_GNN);
std_GCN=std(acc_GCN);
acc_GCN=mean(acc_GCN);
t_GCN=mean(t_GCN);

accN=[acc_AEE_NN,acc_AEE_LDA,acc_ASE_NN,acc_ASE_LDA,acc_LEE_NN,acc_LEE_LDA,acc_LSE_NN,acc_LSE_LDA,acc_GCN,acc_GNN,acc_GFN];
stdN=[std_AEE_NN,std_AEE_LDA,std_ASE_NN,std_ASE_LDA,std_LEE_NN,std_LEE_LDA,std_LSE_NN,std_LSE_LDA,std_GCN,std_GNN,std_GFN];
time=[t_AEE_NN,t_AEE_LDA,t_ASE_NN,t_ASE_LDA,t_LEE_NN,t_LEE_LDA,t_LSE_NN,t_LSE_LDA,t_GCN,t_GNN,t_GFN];

result = array2table([accN; 1-accN; stdN; time], 'RowNames', {'acc', 'err','std', 'time'},'VariableNames', {'AEE_NN', 'AEE_LDA','ASE_NN', 'ASE_LDA','LEE_NN', 'LEE_LDA','LSE_NN', 'LSE_LDA','Kipf_GCN','GNN','GFN'});
