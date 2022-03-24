function simFusion

% Figure 1 SBM
G11_acc_GFN=0; G12_acc_GFN=0; G13_acc_GFN=0; G21_acc_GFN=0; G22_acc_GFN=0;G23_acc_GFN=0;G3_acc_GFN=0;
G11_t_GFN=0; G12_t_GFN=0; G13_t_GFN=0; G21_t_GFN=0; G22_t_GFN=0;G23_t_GFN=0;G3_t_GFN=0;rep=30;
opts = struct('ASE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',1,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
for i=1:10
    for r=1:rep
        n=100*i;
        [Dis,Label]=simGenerate(18,n,1);
        indices = crossvalind('Kfold',Label,10);
        opts.indices=indices;
        result=GraphEncoderEvaluate(Dis(:,:,1),Label,opts);
        G11_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate(Dis(:,:,2),Label,opts);
        G12_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate(Dis(:,:,3),Label,opts);
        G13_acc_GFN(i,r)=result{'acc','GFN'};
        X=zeros(n,n,2);
        X(:,:,1)=Dis(:,:,1);X(:,:,2)=Dis(:,:,3);
        result=GraphEncoderEvaluate(X,Label,opts);
        G21_acc_GFN(i,r)=result{'acc','GFN'};
        X(:,:,1)=Dis(:,:,2);
        result=GraphEncoderEvaluate(X,Label,opts);
        G22_acc_GFN(i,r)=result{'acc','GFN'};
        X(:,:,2)=Dis(:,:,1);
        result=GraphEncoderEvaluate(X,Label,opts);
        G23_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate(Dis,Label,opts);
        G3_acc_GFN(i,r)=result{'acc','GFN'};
    end
end
G11_acc_GFN=mean(G11_acc_GFN,2);
G12_acc_GFN=mean(G12_acc_GFN,2);
G13_acc_GFN=mean(G13_acc_GFN,2);
G21_acc_GFN=mean(G21_acc_GFN,2);
G22_acc_GFN=mean(G22_acc_GFN,2);
G23_acc_GFN=mean(G23_acc_GFN,2);
G3_acc_GFN=mean(G3_acc_GFN,2);


opts = struct('ASE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
opts = struct('ASE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
load('Wiki_Data.mat')
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
opts2=opts;
opts2.deg=1;opts2.ASE=0;
WikiTE=GraphEncoderEvaluate(TE,Label,opts);
WikiTF=GraphEncoderEvaluate(TF,Label,opts);
WikiGE=GraphEncoderEvaluate(GE,Label,opts);
WikiGF=GraphEncoderEvaluate(GF,Label,opts);
WikiTE2=GraphEncoderEvaluate(TE,Label,opts2);
WikiTF2=GraphEncoderEvaluate(TF,Label,opts2);
WikiGE2=GraphEncoderEvaluate(GE,Label,opts2);
WikiGF2=GraphEncoderEvaluate(GF,Label,opts2);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=TE;X(:,:,2)=TF;
WikiText=GraphEncoderEvaluate(X,Label,opts);
WikiText2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=GE;X(:,:,2)=GF;
WikiGraph=GraphEncoderEvaluate(X,Label,opts);
WikiGraph2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(TE,1),size(GE,2),2);
X(:,:,1)=TE;X(:,:,2)=GE;
WikiEng=GraphEncoderEvaluate(X,Label,opts);
WikiEng2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=TF;X(:,:,2)=GF;
WikiFre=GraphEncoderEvaluate(X,Label,opts);
WikiFre2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(TE,1),size(TE,2),4);
X(:,:,1)=TE;X(:,:,2)=TF;X(:,:,3)=GE;X(:,:,4)=GF;
WikiAll=GraphEncoderEvaluate(X,Label,opts);
WikiAll2=GraphEncoderEvaluate(X,Label,opts2);

opts = struct('ASE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
load('graphCElegans.mat')
indices = crossvalind('Kfold',vcols,10);
opts.indices=indices;
opts2=opts;
opts2.deg=1;opts2.ASE=0;
CEAc=GraphEncoderEvaluate(Ac,vcols,opts);
CEAg=GraphEncoderEvaluate(Ag,vcols,opts);
CEAc2=GraphEncoderEvaluate(Ac,vcols,opts2);
CEAg2=GraphEncoderEvaluate(Ag,vcols,opts2);
X=zeros(size(Ac,1),size(Ac,2),2);
X(:,:,1)=Ac;X(:,:,2)=Ag;
CEAll=GraphEncoderEvaluate(X,vcols,opts);
CEAll2=GraphEncoderEvaluate(X,vcols,opts2);

load('pubmed.mat')
load('cora.mat')


opts = struct('ASE',1,'LDA',1,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
load('BrainHippoShape.mat')
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
opts2=opts;
opts2.deg=1;opts2.ASE=0;
BrainLML=GraphEncoderEvaluate(LML,Label,opts);
BrainLMR=GraphEncoderEvaluate(LMR,Label,opts);
BrainSML=GraphEncoderEvaluate(SML,Label,opts);
BrainSMR=GraphEncoderEvaluate(SMR,Label,opts);
BrainLML2=GraphEncoderEvaluate(LML,Label,opts2);
BrainLMR2=GraphEncoderEvaluate(LMR,Label,opts2);
BrainSML2=GraphEncoderEvaluate(SML,Label,opts2);
BrainSMR2=GraphEncoderEvaluate(SMR,Label,opts2);
X=zeros(size(LML,1),size(LML,2),2);
X(:,:,1)=LML;X(:,:,2)=LMR;
BrainL=GraphEncoderEvaluate(X,Label,opts);
BrainL2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(LML,1),size(LML,2),2);
X(:,:,1)=SML;X(:,:,2)=SMR;
BrainS=GraphEncoderEvaluate(X,Label,opts);
BrainS2=GraphEncoderEvaluate(X,Label,opts2);
X=zeros(size(LML,1),size(LML,2),4);
X(:,:,1)=LML;X(:,:,2)=LMR;X(:,:,3)=SML;X(:,:,4)=SMR;
BrainAll=GraphEncoderEvaluate(X,Label,opts);
BrainAll2=GraphEncoderEvaluate(X,Label,opts2);



%%%fusion sim
n=1000;k=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',20,'epoch',100,'training',0.2,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(18,n);
indices = crossvalind('Kfold',Y,5);
opts.indices=indices; opts.Learner=2;opts.LearnIter=0;
% opts2=opts;opts2.Learner=1;opts2.LearnIter=20;
SBM1=GraphEncoderEvaluate(Adj{1},Y,opts);
SBM2=GraphEncoderEvaluate(Adj{2},Y,opts);
SBM3=GraphEncoderEvaluate(Adj{3},Y,opts);
opts.Spectral=0;
SBM12=GraphEncoderEvaluate(Adj(1:2),Y,opts);
SBM23=GraphEncoderEvaluate(Adj(2:3),Y,opts);
SBM13=GraphEncoderEvaluate({Adj{1},Adj{3}},Y,opts);
SBM123=GraphEncoderEvaluate(Adj,Y,opts);

%%% Distance and Kernel
load('Wiki_Data.mat')
Label=Label+1;
% [Z,W]=GraphEncoder(TE,Label,knum); %0 
% [Z,W]=GraphEncoder(TF,Label,knum); %0
% [Z,W]=GraphEncoder(GEAdj,Label,knum); %1138
% [Z,W]=GraphEncoder(GFAdj,Label,knum); %1004
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',5,'epoch',100,'training',0.5,'activation','poslin'); % default parameters
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;opts.Learner=2;opts.LearnIter=0; %opts2.indices=indices;opts2.Learner=1;opts2.LearnIter=20;
WikiTE=GraphEncoderEvaluate(TE,Label,opts);
WikiTF=GraphEncoderEvaluate(TF,Label,opts);
% D=diag(sum(GEAdj,1));
% GEAdj=D^-0.5*(GEAdj+eye(size(GEAdj,1)))*D^-0.5;
WikiGE=GraphEncoderEvaluate(GEAdj,Label,opts);
WikiGF=GraphEncoderEvaluate(GFAdj,Label,opts);
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
% WikiTE2=GraphEncoderEvaluate(TE,Label,opts2);
% WikiTF2=GraphEncoderEvaluate(TF,Label,opts2);
% % % D=diag(sum(GEAdj,1));
% % % GEAdj=D^-0.5*(GEAdj+eye(size(GEAdj,1)))*D^-0.5;
% WikiGE2=GraphEncoderEvaluate(GEAdj,Label,opts2);
% WikiGF2=GraphEncoderEvaluate(GFAdj,Label,opts2);
%%%%Fusion
opts.Spectral=0;
WikiT=GraphEncoderEvaluate({TE,TF},Label,opts);
WikiG=GraphEncoderEvaluate({GE,GF},Label,opts);
WikiE=GraphEncoderEvaluate({TE,GE},Label,opts);
WikiF=GraphEncoderEvaluate({TF,GF},Label,opts);
WikiAll=GraphEncoderEvaluate({TE,TF,GE,GF},Label,opts);


opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',0,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',5,'epoch',100,'training',0.05,'activation','poslin'); % default parameters
% opts = struct('ASE',0,'LDA',0,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
%%% AEE and AEN:
load('graphCElegans.mat')
% knum=1; % all sim models have zero significant node. 
% [Z,W]=GraphEncoder(Ac,vcols,knum); %3 significant nodes
% GraphEncoder(Ag,vcols,knum); %2
indices = crossvalind('Kfold',vcols,5);
opts.indices=indices; opts.Learner=2;
CEAc=GraphEncoderEvaluate(Ac,vcols,opts);
CEAg=GraphEncoderEvaluate(Ag,vcols,opts);
opts.Spectral=0;opts.LearnIter=0;
CE=GraphEncoderEvaluate({Ac,Ag},vcols,opts);

load('CoraAdj.mat') %AEL / GFN K=2
GraphClusteringEvaluate(AdjOri,YOri)
load('email.mat') %k=42
GraphClusteringEvaluate(AdjOri,YOri)
load('lastfm.mat') %AEK K=17
GraphClusteringEvaluate(AdjOri,YOri)
load('polblogs.mat') %k=2
GraphClusteringEvaluate(AdjOri,YOri)
