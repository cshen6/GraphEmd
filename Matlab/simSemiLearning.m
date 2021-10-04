function simSemiLearning

indices = crossvalind('Kfold',Y,num);
trn = (indices == 1); tsn = ~trn;
mdl3=fitsemiself(X(trn,:),Y(trn), X(tsn,:),'Learner','linear','IterationLimit',100);
error3 = mean(Y(tsn)~=mdl3.FittedLabels);



%%% Clustering 
n=10000;k=3;
[Adj,Y]=simGenerate(10,n,k);
simClusteringReal(Adj,Y);
% n=10000;k=10;
% [Adj,Y]=simGenerate(10,n,k);
% simClusteringReal(Adj,Y);

%%% running time
rep=50;nn=1000;ln=20;
opts = struct('Laplacian',0);
%% GCN paramers
num_epoch = 30;        % Number of epochs
d2 = 10;               % Number of hidden units
learning_rate = 1e-4;  % The alpha parameter in the ADAM optimizer
l2_reg = 0;            % L2 regularization weight
batch_size = [];      % Batch size. If empty, equivalent to GCN w/o batching
sample_size = [];     % Sample size. If empty, equivalent to batched GCN

t1=zeros(ln,rep);t2=zeros(ln,rep);t3=zeros(ln,rep);t4=zeros(ln,rep);
for i=1:ln;
    i
    for r=1:rep;
        n=i*nn;
        szW0 = [n,d2];       % Size of parameter matrix W0
        szW1 = [d2,3];       % Size of parameter matrix W1
        num_var = prod(szW0) + prod(szW1);
        adam_param = adam_init(num_var, learning_rate);
        %
      [Adj,Y]=simGenerate(20,n);
      Edge=adj2edge(Adj);
      
      tic
      GraphEncoder(Adj,Y,opts);
      t1(i,r)=toc;
      tic
      GraphEncoder(Edge,Y,opts);
      t2(i,r)=toc;
      
      if opts.Laplacian==0
      Y2=zeros(n,3);
      for j=1:n
          Y2(j,Y(j))=1;
      end
      indices=crossvalind('Kfold',Y,10);
      tsn = (indices == 1); % tst indices
      val = (indices == 2);
      trn2= ~(tsn+val);
      tic
      model_fastgcn_train_and_test(Adj, eye(n), Y2, trn2, val, tsn, ...
            szW0, szW1, l2_reg, num_epoch, batch_size, ...
            sample_size, adam_param);
      t4(i,r)=toc;
      end
      
      tic
      if opts.Laplacian==1
          D=max(sum(Adj,1),1).^(0.5);
          %D=diag(max(sum(Adj,1),1).^-(0.5));
          for j=1:n
              Adj(:,j)=Adj(:,j)/D(j)./D';
          end
      end
      svds(Adj);
      t3(i,r)=toc;
    end
    save(strcat('AEETime',num2str(opts.Laplacian),'.mat'),'t1','t2','t3','t4','rep','nn','ln','opts','i');
end

%%% running time
rep=1;nn=1000;ln=7;
opts = struct('Laplacian',1);
t1=zeros(ln,rep);t2=zeros(ln,rep);
for i=1:ln;
    i
    s=10^(i-1)*nn;
    n=s/100;
    for r=1:rep
        Edge=zeros(s,3);
        Edge(:,1)=randi(n,1,s);
        Edge(:,2)=randi(n,1,s);
        Edge(:,3)=randi(10,1,s);
        Y=randi(10,1,n);
%         [Edge,Y]=simGenerate(20,n,1,1);
%         Edge=adj2edge(Adj);
%         s(i,r)=size(Edge,1);
        
        tic
        GraphEncoder(Edge,Y);
        t1(i,r)=toc;
        tic
        GraphEncoder(Edge,Y,opts);
        t2(i,r)=toc;
        
    end
    save(strcat('AEETimeEdge.mat'),'t1','t2','rep','nn','ln','i');
end

% % %%% Bootstrap
% % n=200; type=11;k=10;
% % [Adj,Y]=simGenerate(type,n,k);
% % opts = struct('indices',crossvalind('Kfold',Y,10),'ASE',1,'LSE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin','resample',0); % default parameters
% % SBM0=GraphEncoderSemi(Adj,Y,opts);
% % % [Adj2,Y2,pval2,stat2]=GraphResample(Adj,Y,n1,0);
% % % [Adj4,Y4,pval2,stat2]=GraphResample(Adj,Y,n2,0);
% % % [Adj3,Y3,pval3,stat3]=GraphResample(Adj,Y,n1,1);
% % n1=3000; [Adj1,Y1]=simGenerate(type,n1);
% % SBM1=GraphEncoderSemi(Adj1,Y1,opts);
% % opts1 = struct('indices',crossvalind('Kfold',Y,10),'ASE',0,'LSE',0,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin','resample',n1); % default parameters
% % SBM2=GraphEncoderSemi(Adj,Y,opts1);
% % n2=5000; opts1.resample=n2;
% % SBM4=GraphEncoderSemi(Adj,Y,opts1);
% % % SBM3=GraphEncoderSemi(Adj3,Y3,opts);
% % 
% % %%
% % load('polblogs.mat'); Y=Label; n1=5000; 
% % load('graphCElegans.mat'); Adj=Ac; Y=vcols; n1=3000; 
% % load('adjnoun.mat');Y=Label; n1=3000; 
% % load('email.mat'); n1=5000; 
% % load('pubmed.mat')
% % load('CoraAdj.mat') %AEK K=7
% % load('Gene.mat');n1=3000;  %AEK K=7
% % opts = struct('indices',crossvalind('Kfold',Y,10),'ASE',1,'LSE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin','resample',0); % default parameters
% % Data1=GraphEncoderSemi(Adj,Y,opts);
% % opts.knn=0;
% % [Adj2,Y2,pval,stat]=GraphResample(Adj,Y,n1,1);
% % Data2=GraphEncoderSemi(Adj2,Y2,opts); 
% % opts.knn=5;[Adj3,Y3,pval,stat]=GraphResample(Adj,Y,n1);
% % %n1=3000; opts1=opts;opts1.resample=n1;
% % Data3=GraphEncoderSemi(Adj3,Y3,opts);
% % opts1 = struct('indices',crossvalind('Kfold',Y,10),'ASE',0,'LSE',0,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin','resample',n1); % default parameters
% % Data4=GraphEncoderSemi(Adj,Y,opts1);

%%% Basic Sims
n=2000;k=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n);
indices = crossvalind('Kfold',Y,5);
opts.indices=indices; 
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
SBM=GraphEncoderSemi(Adj,Y,opts);
% SBM1=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(11,n,k);
SBM2=GraphEncoderSemi(Adj,Y,opts);
% SBM3=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(30,n);
RDPG=GraphEncoderSemi(Adj,Y,opts);
% RDPG1=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(31,n,k);
RDPG2=GraphEncoderSemi(Adj,Y,opts);
% RDPG3=GraphEncoderSemi(Adj,Y,opts2);
% DC-SBM
[Adj,Y]=simGenerate(20,n);
DCSBM=GraphEncoderSemi(Adj,Y,opts);
% DCSBM1=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(21,n,k);
DCSBM2=GraphEncoderSemi(Adj,Y,opts);
% DCSBM3=GraphEncoderSemi(Adj,Y,opts2);
% Dist
opts.LSE=0;opts2.LSE=0;
[Adj,Y]=simGenerate(40,n);
Dist=GraphEncoderSemi(Adj,Y,opts);
% Dist1=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(41,n,k);
Dist2=GraphEncoderSemi(Adj,Y,opts);
% Dist3=GraphEncoderSemi(Adj,Y,opts2);
% Kernel
[Adj,Y]=simGenerate(50,n);
Kern=GraphEncoderSemi(Adj,Y,opts);
% Kern1=GraphEncoderSemi(Adj,Y,opts2);
[Adj,Y]=simGenerate(51,n,k);
Kern2=GraphEncoderSemi(Adj,Y,opts);
% Kern3=GraphEncoderSemi(Adj,Y,opts2);


%%% Repeated Sims
% Figure 1 SBM
SBM_acc_AEE_NN=0;DCSBM_acc_AEE_NN=0; RDPG_acc_AEE_NN=0; SBM_t_AEE_NN=0; DCSBM_t_AEE_NN=0;RDPG_t_AEE_NN=0;
SBM_acc_AEE_LDA=0;DCSBM_acc_AEE_LDA=0; RDPG_acc_AEE_LDA=0; SBM_t_AEE_LDA=0; DCSBM_t_AEE_LDA=0;RDPG_t_AEE_LDA=0;
SBM_acc_ASE_NN=0;DCSBM_acc_ASE_NN=0; RDPG_acc_ASE_NN=0; SBM_t_ASE_NN=0; DCSBM_t_ASE_NN=0;RDPG_t_ASE_NN=0;
SBM_acc_ASE_LDA=0;DCSBM_acc_ASE_LDA=0; RDPG_acc_ASE_LDA=0; SBM_t_ASE_LDA=0; DCSBM_t_ASE_LDA=0;RDPG_t_ASE_LDA=0;
SBM_acc_LSE_NN=0;DCSBM_acc_LSE_NN=0; RDPG_acc_LSE_NN=0; SBM_t_LSE_NN=0; DCSBM_t_LSE_NN=0;RDPG_t_LSE_NN=0;
SBM_acc_LSE_LDA=0;DCSBM_acc_LSE_LDA=0; RDPG_acc_LSE_LDA=0; SBM_t_LSE_LDA=0; DCSBM_t_LSE_LDA=0;RDPG_t_LSE_LDA=0;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
rep=100;num=20; 
for i=1:num
    i
    for r=1:rep
        n=100*i;
        [Adj,Y]=simGenerate(10,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        SBM_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};SBM_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};SBM_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};SBM_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};SBM_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};SBM_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        SBM_t_AEE_NN(i,r)=result{'time','AEE_NN'};SBM_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};SBM_t_ASE_NN(i,r)=result{'time','ASE_NN'};SBM_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};SBM_t_LSE_NN(i,r)=result{'time','LSE_NN'};SBM_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
        [Adj,Y]=simGenerate(20,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        DCSBM_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};DCSBM_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};DCSBM_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};DCSBM_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};DCSBM_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};DCSBM_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        DCSBM_t_AEE_NN(i,r)=result{'time','AEE_NN'};DCSBM_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};DCSBM_t_ASE_NN(i,r)=result{'time','ASE_NN'};DCSBM_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};DCSBM_t_LSE_NN(i,r)=result{'time','LSE_NN'};DCSBM_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
        [Adj,Y]=simGenerate(30,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        RDPG_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};RDPG_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};RDPG_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};RDPG_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};RDPG_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};RDPG_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        RDPG_t_AEE_NN(i,r)=result{'time','AEE_NN'};RDPG_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};RDPG_t_ASE_NN(i,r)=result{'time','ASE_NN'};RDPG_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};RDPG_t_LSE_NN(i,r)=result{'time','LSE_NN'};RDPG_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
    end
end

% Figure 2 SBM
clear
SBM_acc_AEE_NN=0;DCSBM_acc_AEE_NN=0; RDPG_acc_AEE_NN=0; SBM_t_AEE_NN=0; DCSBM_t_AEE_NN=0;RDPG_t_AEE_NN=0;
SBM_acc_AEE_LDA=0;DCSBM_acc_AEE_LDA=0; RDPG_acc_AEE_LDA=0; SBM_t_AEE_LDA=0; DCSBM_t_AEE_LDA=0;RDPG_t_AEE_LDA=0;
SBM_acc_ASE_NN=0;DCSBM_acc_ASE_NN=0; RDPG_acc_ASE_NN=0; SBM_t_ASE_NN=0; DCSBM_t_ASE_NN=0;RDPG_t_ASE_NN=0;
SBM_acc_ASE_LDA=0;DCSBM_acc_ASE_LDA=0; RDPG_acc_ASE_LDA=0; SBM_t_ASE_LDA=0; DCSBM_t_ASE_LDA=0;RDPG_t_ASE_LDA=0;
SBM_acc_LSE_NN=0;DCSBM_acc_LSE_NN=0; RDPG_acc_LSE_NN=0; SBM_t_LSE_NN=0; DCSBM_t_LSE_NN=0;RDPG_t_LSE_NN=0;
SBM_acc_LSE_LDA=0;DCSBM_acc_LSE_LDA=0; RDPG_acc_LSE_LDA=0; SBM_t_LSE_LDA=0; DCSBM_t_LSE_LDA=0;RDPG_t_LSE_LDA=0;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
rep=100;num=10;
for i=1:num
    i
    for r=1:rep
        n=200*i;
        [Adj,Y]=simGenerate(11,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        SBM_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};SBM_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};SBM_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};SBM_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};SBM_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};SBM_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        SBM_t_AEE_NN(i,r)=result{'time','AEE_NN'};SBM_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};SBM_t_ASE_NN(i,r)=result{'time','ASE_NN'};SBM_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};SBM_t_LSE_NN(i,r)=result{'time','LSE_NN'};SBM_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
        [Adj,Y]=simGenerate(21,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        DCSBM_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};DCSBM_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};DCSBM_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};DCSBM_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};DCSBM_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};DCSBM_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        DCSBM_t_AEE_NN(i,r)=result{'time','AEE_NN'};DCSBM_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};DCSBM_t_ASE_NN(i,r)=result{'time','ASE_NN'};DCSBM_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};DCSBM_t_LSE_NN(i,r)=result{'time','LSE_NN'};DCSBM_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
        [Adj,Y]=simGenerate(31,n);
        result=GraphEncoderSemi(Adj,Y,opts);
        RDPG_acc_AEE_NN(i,r)=result{'acc','AEE_NN'};RDPG_acc_AEE_LDA(i,r)=result{'acc','AEE_LDA'};RDPG_acc_ASE_NN(i,r)=result{'acc','ASE_NN'};RDPG_acc_ASE_LDA(i,r)=result{'acc','ASE_LDA'};RDPG_acc_LSE_NN(i,r)=result{'acc','LSE_NN'};RDPG_acc_LSE_LDA(i,r)=result{'acc','LSE_LDA'};
        RDPG_t_AEE_NN(i,r)=result{'time','AEE_NN'};RDPG_t_AEE_LDA(i,r)=result{'time','AEE_LDA'};RDPG_t_ASE_NN(i,r)=result{'time','ASE_NN'};RDPG_t_ASE_LDA(i,r)=result{'time','ASE_LDA'};RDPG_t_LSE_NN(i,r)=result{'time','LSE_NN'};RDPG_t_LSE_LDA(i,r)=result{'time','LSE_LDA'};
    end
end

%%% Real Data
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',0,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
% opts = struct('ASE',0,'LDA',0,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
%%% AEE and AEN:
load('graphCElegans.mat')
% knum=1; % all sim models have zero significant node. 
% [Z,W]=GraphEncoder(Ac,vcols,knum); %3 significant nodes
% GraphEncoder(Ag,vcols,knum); %2
indices = crossvalind('Kfold',vcols,10);
opts.indices=indices; 
CEAc=GraphEncoderSemi(Ac,vcols,opts);
CEAg=GraphEncoderSemi(Ag,vcols,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% CEAc2=GraphEncoderSemi(Ac,vcols,opts2);
% CEAg2=GraphEncoderSemi(Ag,vcols,opts2);

load('adjnoun.mat')
%knum=3; % all sim models have zero significant node. 
%[Z,W]=GraphEncoder(Adj,Label,knum); %15
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
AN=GraphEncoderSemi(Adj,Label,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% AN2=GraphEncoderSemi(Adj,Label,opts2);

%%% AEN only:
load('polblogs.mat') 
% GraphEncoder(Adj,Label,knum); %8
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
PB=GraphEncoderSemi(Adj,Label,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% PB2=GraphEncoderSemi(Adj,Label,opts2);

%%% Distance and Kernel
load('Wiki_Data.mat')
% [Z,W]=GraphEncoder(TE,Label,knum); %0 
% [Z,W]=GraphEncoder(TF,Label,knum); %0
% [Z,W]=GraphEncoder(GEAdj,Label,knum); %1138
% [Z,W]=GraphEncoder(GFAdj,Label,knum); %1004
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
WikiTE=GraphEncoderSemi(TE,Label,opts);
WikiTF=GraphEncoderSemi(TF,Label,opts);
% D=diag(sum(GEAdj,1));
% GEAdj=D^-0.5*(GEAdj+eye(size(GEAdj,1)))*D^-0.5;
WikiGE=GraphEncoderSemi(GEAdj,Label,opts);
WikiGF=GraphEncoderSemi(GFAdj,Label,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% WikiTE2=GraphEncoderSemi(TE,Label,opts2);
% WikiTF2=GraphEncoderSemi(TF,Label,opts2);
% % D=diag(sum(GEAdj,1));
% % GEAdj=D^-0.5*(GEAdj+eye(size(GEAdj,1)))*D^-0.5;
% WikiGE2=GraphEncoderSemi(GEAdj,Label,opts2);
% WikiGF2=GraphEncoderSemi(GFAdj,Label,opts2);
% 
% 
% 
% %%% Pending
% load('Wiki_Data.mat')
% opts = struct('neuron',30,'epoch',100,'training',0.8); % default parameters
% [GE_acc_AEE,GE_acc_AEE2,GE_acc_AEN,GE_acc_ASE,GE_t_AEE,GE_t_AEE2,GE_t_AEN,GE_t_ASE]=GraphEncoderSemi(GEAdj,Label,opts);
% [GF_acc_AEE,GF_acc_AEE2,GF_acc_AEN,GF_acc_ASE,GF_t_AEE,GF_t_AEE2,GF_t_AEN,GF_t_ASE]=GraphEncoderSemi(GFAdj,Label,opts);

%%% when k is large relative to n, GCN overfit???

load('CoraAdj.mat') %AEK K=7
% GraphEncoder(Adj,Y,knum); %0 
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
Cora=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% Cora2=GraphEncoderSemi(Adj,Y,opts2);

% load('DD244.mat') %GCN K=20
% % GraphEncoder(Adj,Y,knum); %249 
% DD=GraphEncoderSemi(Adj,Y,opts);

load('Gene.mat') %AEL / GFN K=2
% GraphEncoder(Adj,Y,knum); %0
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
Gene=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% Gene2=GraphEncoderSemi(Adj,Y,opts2);

% load('KKI.mat')%GCN K=189
% % GraphEncoder(Adj,Y,knum); %2238
% KKI=GraphEncoderSemi(Adj,Y,opts);
load('IIP.mat') %AEL / GFN K=2
% GraphEncoder(Adj,Y,knum); %0
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
IIP=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% IIP2=GraphEncoderSemi(Adj,Y,opts2);

load('lastfm.mat') %AEK K=17
% GraphEncoder(Adj,Y,knum); %542
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
LFM=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% LFM2=GraphEncoderSemi(Adj,Y,opts2);

% load('OHSU.mat')%GCN K=189
% % GraphEncoder(Adj,Y,knum); %6479 
% OHSU=GraphEncoderSemi(Adj,Y,opts);
% 
% load('Peking.mat')%GCN K=189
% % GraphEncoder(Adj,Y,knum); %3341
% Pek=GraphEncoderSemi(Adj,Y,opts);

load('pubmed.mat')
% GraphEncoder(Adj,Y,knum); %7 
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
Pub=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% Pub2=GraphEncoderSemi(Adj,Y,opts2);

load('email.mat')
indices = crossvalind('Kfold',Y,10);
opts.indices=indices;
email=GraphEncoderSemi(Adj,Y,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% email2=GraphEncoderSemi(Adj,Y,opts2);
% %%% GCN False Example
% n=2000;k=5;p=0.4;
% Y=randi(k,n,1);
% X=unifrnd(0,1,n,n);
% X=(X+X')/2;
% X=double(X<p);
% for i=1:n
% X(i,i)=0;
% end
% [acc_AEL,t_AEL, acc_GFN, t_GFN, acc_ASE, t_ASE, acc_GCN,t_GCN, acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderSemi(X,Y);