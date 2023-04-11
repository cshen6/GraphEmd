function [result]=GraphEncoderEvaluate(X,Y,opts,D)

if nargin < 4
    D=0;
end
if nargin < 3
    opts = struct('eval',1,'indices',crossvalind('Kfold',Y,10),'Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30,'neuron',20,'epoch',100,'training',0.05,'activation','poslin','Elbow',0); % default parameters
end
if ~isfield(opts,'eval'); opts.eval=1; end
if ~isfield(opts,'indices'); opts.indices=crossvalind('Kfold',Y,10); end
if ~isfield(opts,'Adjacency'); opts.Adjacency=1; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=0; end
if ~isfield(opts,'Spectral'); opts.Spectral=0; end % 1 for ASE and Omni; 2 for USE; 3 for MASE
if ~isfield(opts,'LDA'); opts.LDA=0; end
if ~isfield(opts,'GNN'); opts.GNN=0; end
if ~isfield(opts,'knn'); opts.knn=5; end
if ~isfield(opts,'dim'); opts.dim=30; end
% if ~isfield(opts,'deg'); opts.deg=0; end
if ~isfield(opts,'neuron'); opts.neuron=20; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'training'); opts.training=0.05; end
if ~isfield(opts,'activation'); opts.activation='poslin'; end %purelin, tansig
if ~isfield(opts,'Elbow'); opts.Elbow=0; end
warning('off','all');
%met=[opts.AEE,opts.LDA,opts.GFN,opts.ASE,opts.LSE,opts.GCN,opts.GNN]; %AEE, LDA, GFN, ASE, GFN, ANN
indices=opts.indices;
% if length(indices)~=length(Y)
%     indices=crossvalind('Kfold',Y,10);
% end
if iscell(Y)
    Y2=Y;
    Y=Y{1,1};
else
    Y2=0;
end
kfold=max(indices);
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

acc_AEE_NN=zeros(kfold,1);acc_AEE_LDA=zeros(kfold,1);acc_GNN=zeros(kfold,1);
acc_LEE_NN=zeros(kfold,1);acc_LEE_LDA=zeros(kfold,1);t_LEE_NN=zeros(kfold,1);t_LEE_LDA=zeros(kfold,1);
t_AEE_NN=zeros(kfold,1);t_AEE_LDA=zeros(kfold,1);t_GNN=zeros(kfold,1);
acc_ASE_NN=zeros(kfold,d);t_ASE_NN=zeros(kfold,d);acc_ASE_LDA=zeros(kfold,d);t_ASE_LDA=zeros(kfold,d);acc_LSE_NN=zeros(kfold,d);t_LSE_NN=zeros(kfold,d);acc_LSE_LDA=zeros(kfold,d);t_LSE_LDA=zeros(kfold,d);
acc_GCN=zeros(kfold,1);t_GCN=zeros(kfold,1);
% opts1 = struct('deg',opts.deg); % default parameters
% opts2 = struct('deg',opts.deg,'pivot',opts.pivot); % default parameters
if opts.Spectral>0
    % ASE
    numK=1;
    if iscell(X)
        numK=length(X);
        if opts.Spectral==1
            Adj=Omni(X);
        else
            Adj=zeros(n,n*numK);
            for ii=1:numK
                if size(X{ii},2)<=3
                    Adj(:,(ii-1)*n+1:(ii-1)*n+n)=edge2adj(X{ii});
                else
                    Adj(:,(ii-1)*n+1:(ii-1)*n+n)=X{ii};
                end
            end
        end
        if opts.Spectral==3
            dimS=30;
            Adj2=zeros(n,dimS*numK);
            for ii=1:numK
                [U,S,~]=svds(Adj(:,(ii-1)*n+1:(ii-1)*n+n),dimS);
                Adj2(:,(ii-1)*dimS+1:(ii-1)*dimS+dimS)=U(:,1:dimS)*S(1:dimS,1:dimS)^0.5;
            end
            Adj=Adj2;
        end
    else
        if size(X,2)<=3
            Adj=edge2adj(X);
        else
            Adj=X;
        end
    end
end

for i = 1:kfold
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
            YT(tsn)=0;
            YTrn=Y(trn);
            YTsn=Y(tsn);
%             size(Y)
            tic
            oot=opts;
            oot.Laplacian=0; 
            if ~iscell(Y2)
                if opts.eval==1
                   Z=GraphEncoder(X,YT,D,oot);
                else
                   Z=GraphEncoder(X,0,D,oot);
                end
            else
                Y3=Y2;
                Y3{1}(tsn)=0;
                Z=GraphEncoder(X,Y3,D,oot);
            end
            if iscell(Z)
                Z=horzcat(Z{:});
            end
%             if opts.Elbow>0
%                 stdZ=std(Z);
%                 [stdZ2,dimInd]=sort(stdZ,'descend');
%                 if (stdZ2(1)-stdZ2(end))/stdZ2(1)>0.1
%                     [idx,center]=kmeans(stdZ',2);
%                     dimInd=(idx==1);
%                     if center(2)>center(1)
%                         dimInd=~dimInd;
%                     end
%                     %         q=getElbow(stdZ,1)
%                     Z=Z(:,dimInd);
%                 end
%             end
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
            
            if opts.eval==1
            if opts.knn>0
                tic
                mdl=fitcknn(ZTrn,YTrn,'Distance','Euclidean','NumNeighbors',opts.knn);
                tt=predict(mdl,ZTsn);
                t_AEE_NN(i)=tmp1+toc;
                acc_AEE_NN(i)=acc_AEE_NN(i)+mean(YTsn~=tt);
            end
            
            if opts.LDA==1
                tic
                mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
                tt=predict(mdl,ZTsn);
                t_AEE_LDA(i)=tmp1+toc;
                acc_AEE_LDA(i)=acc_AEE_LDA(i)+mean(YTsn~=tt);
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
                t_GNN(i)=tmp1+toc;
                acc_GNN(i)=acc_GNN(i)+mean(YTsn~=tt);
            end
            end
            if opts.eval==2
                tic
%                 if D==0
%                    mdl=fitlm(ZTrn,YTrn);
%                    tt=predict(mdl,ZTsn);
%                 else
                   %mdl=fitlm([ZTrn,D(trn,:)],YTrn,'NumNeighbors',opts.knn);
                   mdl=fitrensemble([ZTrn,D(trn,:)],YTrn);
%                    mdl=fitrnet([ZTrn,D(trn,:)],YTrn);
                   tt=predict(mdl,[ZTsn,D(tsn,:)]);
%                 end
                err =sum((YTsn-tt).^2)/sum((YTsn-mean(YTsn)).^2);
                t_AEE_NN(i)=tmp1+toc;
                acc_AEE_NN(i)=acc_AEE_NN(i)+err;
            end
            
            if opts.Spectral>0
                % ASE
                YA=Y;
                tic
                %         if nre>n
                %             Adj=AdjRe;
                %             trn=1:nre;
                %             tsn=nre+1:size(AdjRe,1);
                %             YA=YRe;
                %         end
%                 if opts.Spectral~=3
                    [U,S,V]=svds(Adj,d);
                    if opts.Spectral==2
                        U=V;
                    end
%                 end
                t1=toc;
                for j=1:d
                    tic
                    if opts.Spectral~=3
                        Z=U(:,1:j)*S(1:j,1:j)^0.5;
                    else
                        Z=U(:,1:j);
                    end
                    if opts.Spectral==2
                       Z=reshape(Z,n,j*numK);
                    end
                    t2=toc;
                    if opts.eval==1;
                    if opts.LDA==1
                        tic
                        mdl=fitcdiscr(Z(trn,:),YA(trn),'DiscrimType',discrimType);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_LDA(i,j)=t1+t2;%+toc;
                        acc_ASE_LDA(i,j)=acc_ASE_LDA(i,j)+mean(YTsn~=tt);
                    end
                    if opts.knn>0
                        tic
                        mdl=fitcknn(Z(trn,:),YA(trn),'Distance','euclidean','NumNeighbors',opts.knn);
                        tt=predict(mdl,Z(tsn,:));
                        t_ASE_NN(i,j)=t1+t2;%+toc;
                        acc_ASE_NN(i,j)=acc_ASE_NN(i,j)+mean(YTsn~=tt);
                    end
                    end
                    if opts.eval==2
                        tic
%                         if D==0
%                             mdl=fitlm(Z(trn,:),YA(trn));
%                             tt=predict(mdl,Z(tsn,:));
%                         else
                            mdl=fitrensemble([Z(trn,:),D(trn,:)],YA(trn));
                            tt=predict(mdl,[Z(tsn,:),D(tsn,:)]);
%                         end
                        err =sum((YTsn-tt).^2)/sum((YTsn-mean(YTsn)).^2);
                        t_ASE_NN(i,j)=t1+t2;
                        acc_ASE_NN(i,j)=acc_ASE_NN(i,j)+err;
                    end
                end
            end
        end
        
%         if opts.Laplacian==1
%             YT=Y;
%             YT(tsn)=-1;
%             oot=opts;
%             oot.Laplacian=1; 
% %             oot=struct('Laplacian',true,'LearnIter',0,'Learner',opts.Learner,'Dim',opts.dimGEE);
%             YTrn=Y(trn);
%             YTsn=Y(tsn);
%             tic
%             Z=GraphEncoder(X,YT,D,oot);
%             if iscell(Z)
%                 Z=cell2mat(Z');
%             end
%             ZTrn=Z(trn,:);
%             ZTsn=Z(tsn,:);
%             %     else
%             %         [Z,indT]=GraphEncoder(X,YT,opts);
%             %         %[Z,indT]=GraphSBMEst(X,YT);
%             %     end
%             %     if k>klim
%             %         [ind,~,~] = DCorScreening(Z,Y(trn));
%             %         Z=Z(:,ind);
%             %     end
%             tmp1=toc;
%             if opts.knn>0
%                 tic
%                 mdl=fitcknn(ZTrn,YTrn,'Distance','Euclidean','NumNeighbors',opts.knn);
%                 tt=predict(mdl,ZTsn);
%                 t_LEE_NN(i)=tmp1;
%                 acc_LEE_NN(i)=acc_LEE_NN(i)+mean(YTsn~=tt);
%             end
%             
%             if opts.LDA==1
%                 tic
%                 mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
%                 tt=predict(mdl,ZTsn);
%                 t_LEE_LDA(i)=tmp1;
%                 acc_LEE_LDA(i)=acc_LEE_LDA(i)+mean(YTsn~=tt);
%             end
%             
%             
%             % ASE
%             if opts.Spectral==1
%                 tic
%                 Adj=Omni(X);
%                 D=max(sum(Adj,1),1).^(0.5);
%                 AdjT=Adj;
%                 for j=1:n
%                     AdjT(:,j)=AdjT(:,j)/D(j)./D';
%                 end
%                 [U,S,~]=svds(AdjT,d);
%                 t1=toc;
%                 for j=1:d
%                     tic
%                     Z=U(:,1:j)*S(1:j,1:j)^0.5;
%                     t2=toc;
%                     if opts.LDA==1
%                         tic
%                         mdl=fitcdiscr(Z(trn,:),Y(trn),'DiscrimType',discrimType);
%                         tt=predict(mdl,Z(tsn,:));
%                         t_LSE_LDA(i,j)=t1+t2;
%                         acc_LSE_LDA(i,j)=acc_LSE_LDA(i,j)+mean(Y(tsn)~=tt);
%                     end
%                     if opts.knn>0
%                         tic
%                         mdl=fitcknn(Z(trn,:),Y(trn),'Distance','euclidean','NumNeighbors',opts.knn);
%                         tt=predict(mdl,Z(tsn,:));
%                         t_LSE_NN(i,j)=t1+t2;
%                         acc_LSE_NN(i,j)=acc_LSE_NN(i,j)+mean(Y(tsn)~=tt);
%                     end
%                 end
%             end
%         end
        
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
end

% [~,ind]=min(mean(acc_ASE_NN,1));
% [h,p]=ttest(acc_AEE_NN, acc_ASE_NN(:,ind),'Tail','left');
% [h,p]=ttest(t_AEE_NN, t_ASE_NN(:,ind),'Tail','left');

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
% std_GCN=std(acc_GCN);
% acc_GCN=mean(acc_GCN);
% t_GCN=mean(t_GCN);

accN=[acc_GNN, acc_AEE_NN,acc_AEE_LDA,acc_ASE_NN,acc_ASE_LDA, acc_LEE_NN,acc_LEE_LDA,acc_LSE_NN,acc_LSE_LDA];
stdN=[std_GNN, std_AEE_NN,std_AEE_LDA,std_ASE_NN,std_ASE_LDA, std_LEE_NN,std_LEE_LDA,std_LSE_NN,std_LSE_LDA];
time=[t_GNN, t_AEE_NN,t_AEE_LDA,t_ASE_NN,t_ASE_LDA, t_LEE_NN,t_LEE_LDA,t_LSE_NN,t_LSE_LDA];

result = array2table([1-accN; accN; stdN; time], 'RowNames', {'err', 'acc','std', 'time'},'VariableNames', {'GEE_NN', 'AEE_KNN', 'AEE_LDA','ASE_KNN', 'ASE_LDA','LEE_KNN', 'LEE_LDA','LSE_KNN', 'LSE_LDA'});

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