function [result]=GraphEncoderEvaluate2(X,Y,opts,D)

if nargin < 4
    D=0;
end
if nargin < 3
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',1,'knn',5,'dim',30,'neuron',20,'epoch',100,'training',0.05,'activation','poslin'); % default parameters
end
if ~isfield(opts,'Adjacency'); opts.Adjacency=1; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=0; end
if ~isfield(opts,'Spectral'); opts.Spectral=0; end
if ~isfield(opts,'LDA'); opts.LDA=0; end
if ~isfield(opts,'GNN'); opts.GNN=1; end
if ~isfield(opts,'knn'); opts.knn=5; end
if ~isfield(opts,'dim'); opts.dim=30; end
% if ~isfield(opts,'deg'); opts.deg=0; end
if ~isfield(opts,'neuron'); opts.neuron=20; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'training'); opts.training=0.05; end
if ~isfield(opts,'activation'); opts.activation='poslin'; end %purelin, tansig
warning('off','all');
%met=[opts.AEE,opts.LDA,opts.GFN,opts.ASE,opts.LSE,opts.GCN,opts.GNN]; %AEE, LDA, GFN, ASE, GFN, ANN
numTrn=160;
[K,~,Y]=unique(Y);
n=length(Y);
d=min(opts.dim,n-1);
K=length(K);
opts.knn=min(opts.knn,ceil(n/K/3));
discrimType='pseudoLinear';

% initialize NN
if opts.GNN>0
netGNN = patternnet(max(opts.neuron,K),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
netGNN.layers{1}.transferFcn = opts.activation;
netGNN.trainParam.showWindow = false;
netGNN.trainParam.epochs=opts.epoch;
netGNN.divideParam.trainRatio = 0.9;
netGNN.divideParam.valRatio   = 0.1;
netGNN.divideParam.testRatio  = 0/100;
end
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

% %% GCN paramers
% if opts.GCN>0
% num_epoch = 100;        % Number of epochs
% d2 = 10;               % Number of hidden units
% learning_rate = 1e-4;  % The alpha parameter in the ADAM optimizer
% l2_reg = 0;            % L2 regularization weight
% batch_size = [];      % Batch size. If empty, equivalent to GCN w/o batching
% sample_size = [];     % Sample size. If empty, equivalent to batched GCN
% szW0 = [n,d2];       % Size of parameter matrix W0
% szW1 = [d2,K];       % Size of parameter matrix W1
% num_var = prod(szW0) + prod(szW1);
% end
% adam_param = adam_init(num_var, learning_rate);


acc_ASE_NN=zeros(1,d);t_ASE_NN=zeros(1,d);acc_ASE_LDA=zeros(1,d);t_ASE_LDA=zeros(1,d);acc_LSE_NN=zeros(1,d);t_LSE_NN=zeros(1,d);acc_LSE_LDA=zeros(1,d);t_LSE_LDA=zeros(1,d);
% opts1 = struct('deg',opts.deg); % default parameters
% opts2 = struct('deg',opts.deg,'pivot',opts.pivot); % default parameters

trn=zeros(n,1);
for i=1:K
    indTmp=find(Y==i);
    tmp=randperm(length(indTmp));
    trn(indTmp(tmp(1:numTrn)))=1;
end
tsn=~trn;
trn=~tsn;
        %     tst = (indices == i); % tst indices
        %     trn = ~tst; % trning indices
        
%         tsn = (indices == i); % tst indices
%         trn = ~tsn; % trning indices
%         
%         %     trn = (indices == i); % tst indices
%         %     tsn = ~trn; % trning indices
%         
%         val = (indices == max(mod(i+1,tsn+1),1));
        %     trn2= ~(tsn+val);
        
        if opts.Adjacency==1
            YT=Y;
            YT(tsn)=-1;
            YTrn=Y(trn);
            YTsn=Y(tsn);
%             size(Y)
            tic
            oot=opts;
            oot.Laplacian=0; 
            Z=GraphEncoder(X,YT,D,oot);
            if iscell(Z)
                Z=cell2mat(Z');
            end
            ZTrn=Z(trn,:);
            ZTsn=Z(tsn,:);
%             size(Z)
            
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
%                 tic
                mdl=fitcknn(ZTrn,YTrn,'Distance','Euclidean','NumNeighbors',opts.knn);
                tt=predict(mdl,ZTsn);
                t_AEE_NN=tmp1;
                acc_AEE_NN=mean(YTsn~=tt);
            end
            
            if opts.LDA==1
%                 tic
                mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
                tt=predict(mdl,ZTsn);
                t_AEE_LDA=tmp1;
                acc_AEE_LDA=mean(YTsn~=tt);
            end
            
            
            if opts.GNN==1
                tic
                Y2=onehotencode(categorical(YTrn),2)';
%                 Y2=zeros(length(YTrn),K);
%                 for j=1:length(YTrn)
%                     Y2(j,YTrn(j))=1;
%                 end
                %         Y2Trn=Y2(trn,:);
                mdl3 = train(netGNN,ZTrn',Y2);
                classes = mdl3(ZTsn'); % class-wise probability for tsting data
                %acc_NN = perform(mdl3,Y2Tsn',classes);
                tt = vec2ind(classes)'; % this gives the actual class for each observation
                t_GNN=tmp1+toc;
                acc_GNN=mean(YTsn~=tt);
            end
            
            if opts.Spectral==1
                % ASE
                if iscell(X)
                   Adj=Omni(X);
                else
                    if size(X,2)<=3
                        Adj=edge2adj(X);
                    else
                        Adj=X;
                    end
                end
                YA=Y;
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
                        mdl=fitcdiscr(Z(trn,:),YA(trn),'DiscrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_LDA(j)=t1+t2;%+toc;
                        acc_ASE_LDA(j)=acc_ASE_LDA(j)+mean(YTsn~=tt);
                    end
                    if opts.knn>0
                        tic
                        mdl=fitcknn(Z(trn,:),YA(trn),'Distance','euclidean','NumNeighbors',opts.knn);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_NN(j)=t1+t2;%+toc;
                        acc_ASE_NN(j)=acc_ASE_NN(j)+mean(YTsn~=tt);
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
        
%         %  Direct NN
%         if opts.GNN==1
%             tic
%             X1=reshape(X(trn,trn,:),sum(trn),num*sum(trn));
%             X2=reshape(X(tsn,trn,:),sum(tsn),num*sum(trn));
%             mdl3 = train(netGNN,X1',Y2Trn');
%             classes = mdl3(X2'); % class-wise probability for tsting data
%             tt = vec2ind(classes)'; % this gives the actual class for each observation
%             t_GNN(i)=tmp2+toc;
%             acc_GNN(i)=acc_GNN(i)+mean(Y(tsn)~=tt);
%         end

% [~,ind]=min(mean(acc_ASE_NN,1));
% [h,p]=ttest(acc_AEE_NN, acc_ASE_NN(:,ind),'Tail','left');
% [h,p]=ttest(t_AEE_NN, t_ASE_NN(:,ind),'Tail','left');

acc_AEE_NN=1-mean(acc_AEE_NN);
t_AEE_NN=mean(t_AEE_NN);
% acc_AEE_LDA=1-mean(acc_AEE_LDA);
% t_AEE_LDA=mean(t_AEE_LDA);
[acc_ASE_NN,ind]=min(acc_ASE_NN);
acc_ASE_NN=1-acc_ASE_NN;
t_ASE_NN=mean(t_ASE_NN,1); t_ASE_NN=t_ASE_NN(ind);
% [acc_ASE_LDA,ind]=min(acc_ASE_LDA);
% acc_ASE_LDA=1-acc_ASE_LDA;
% 
% acc_GNN=1-mean(acc_GNN);
% t_GNN=mean(t_GNN);
% std_GCN=std(acc_GCN);
% acc_GCN=mean(acc_GCN);
% t_GCN=mean(t_GCN);

accN=[acc_AEE_NN,acc_ASE_NN];
time=[t_AEE_NN,t_ASE_NN,];

result = array2table([accN; 1-accN; time], 'RowNames', {'acc', 'err', 'time'},'VariableNames', {'AEE_KNN','ASE_KNN'});

function A=Omni(A);
if iscell(A)
    numG=length(A);
    for i=1:numG
        if size(A{i},2)<=3
            A{i}=edge2adj(A{i});
        end
    end
    d=size(A{1},1);
    AOm=zeros(d*numG,d*numG);
    for i=1:numG
        AOm((i-1)*d+1:(i-1)*d+d,(i-1)*d+1:(i-1)*d+d)=A{i};
        for j=i+1:numG
            AOm((i-1)*d+1:(i-1)*d+d,(j-1)*d+1:(j-1)*d+d)=(A{i}+A{j})/2;
            AOm((j-1)*d+1:(j-1)*d+d,(i-1)*d+1:(i-1)*d+d)=AOm((i-1)*d+1:(i-1)*d+d,(j-1)*d+1:(j-1)*d+d);
        end
    end
    A=AOm;
else
    return;
end