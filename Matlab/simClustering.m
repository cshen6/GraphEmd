function simClustering

%%% self-learning
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n,K);
indices = crossvalind('Kfold',Y,5);error1=0;
for i=1:max(indices);
trn = (indices == i); % tst indices
tsn = ~trn; % trning indices
Y2=Y;Y2(tsn)=-1;
opts = struct('Laplacian',false,'maxIter',20,'Learn',true);
[Z,Y3,W,indT,B]=GraphEncoder(Adj,Y2,opts);
error1=error1+mean(Y3(tsn)==Y(tsn))/max(indices);
end

%%%%
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(11,n,5);
GraphEncoderC(Adj,5,Y);
[Adj,Y]=simGenerate(20,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(21,n,5);
GraphEncoderC(Adj,5,Y);
[Adj,Y]=simGenerate(30,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(31,n,5);
GraphEncoderC(Adj,5,Y);

%%% Basic Sims
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n);
indices = crossvalind('Kfold',Y,5);
opts.indices=indices; 
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
SBM=GraphEncoderSemi(Adj,Y,opts);
% SBM1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(11,n,K);
SBM2=GraphEncoderSemi(Adj,Y,opts);
% SBM3=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(30,n);
RDPG=GraphEncoderSemi(Adj,Y,opts);
% RDPG1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(31,n,K);
RDPG2=GraphEncoderSemi(Adj,Y,opts);
% RDPG3=GraphEncoderEvaluate(Adj,Y,opts2);
% DC-SBM
[Adj,Y]=simGenerate(20,n);
DCSBM=GraphEncoderSemi(Adj,Y,opts);
% DCSBM1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(21,n,K);
DCSBM2=GraphEncoderSemi(Adj,Y,opts);
% DCSBM3=GraphEncoderEvaluate(Adj,Y,opts2);
% Dist
opts.LSE=0;opts2.LSE=0;
[Adj,Y]=simGenerate(40,n);
Dist=GraphEncoderSemi(Adj,Y,opts);
% Dist1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(41,n,K);
Dist2=GraphEncoderSemi(Adj,Y,opts);
% Dist3=GraphEncoderEvaluate(Adj,Y,opts2);
% Kernel
[Adj,Y]=simGenerate(50,n);
Kern=GraphEncoderSemi(Adj,Y,opts);
% Kern1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(51,n,K);
Kern2=GraphEncoderSemi(Adj,Y,opts);


%%% Basic Sims
n=5000;
% opts = struct('ASE',1,'LSE',1,'NN',0,'Dist','cosine','maxIter',20,'normalize',0,'deg',0,'dmax',30); % default parameters
K=10;rep=10; tt=3;
SBM=zeros(2,5,tt);
RDPG=zeros(2,5,tt);
DCSBM=zeros(2,5,tt);
for r=1:rep
[Adj,Y]=simGenerate(10,n,K);
% % Edge=Adj2Edge(Adj);
SBM(:,:,1)=SBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(11,n,K);
% Edge=Adj2Edge(Adj);
SBM(:,:,2)=SBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(12,n,K);
% Edge=Adj2Edge(Adj);
SBM(:,:,3)=SBM(:,:,3)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% [Adj,Y]=simGenerate(30,n,k);
% % % Edge=Adj2Edge(Adj);
% RDPG(:,:,1)=RDPG(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% [Adj,Y]=simGenerate(31,n,k);
% % Edge=Adj2Edge(Adj);
% RDPG(:,:,2)=RDPG(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% DC-SBM
[Adj,Y]=simGenerate(20,n);
% % Edge=Adj2Edge(Adj);
DCSBM(:,:,1)=DCSBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(21,n,K);
% Edge=Adj2Edge(Adj);
DCSBM(:,:,2)=DCSBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(22,n,K);
% Edge=Adj2Edge(Adj);
DCSBM(:,:,3)=DCSBM(:,:,3)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
end

n=2000;K=3;
[Adj,Y]=simGenerate(10,n,K);
simClusteringReal(Adj,Y);

% Dist
opts.LSE=0;opts2.LSE=0;
[Adj,Y]=simGenerate(40,n);
Dist=GraphClusteringEvaluate(Adj,Y);
[Adj,Y]=simGenerate(41,n,K);
Dist2=GraphClusteringEvaluate(Adj,Y);
% Kernel
[Adj,Y]=simGenerate(50,n);
Kern=GraphClusteringEvaluate(Adj,Y);
[Adj,Y]=simGenerate(51,n,K);
Kern2=GraphClusteringEvaluate(Adj,Y);





n=3000;
[Adj,Y]=simGenerate(1,n);
Dist='sqeuclidean'; %Dist='cosine';
[SBM_RI_AEE,SBM_RI_AEN,SBM_RI_ASE,SBM_t_AEE,SBM_t_AEN,SBM_t_ASE,SBM_ind_AEE,SBM_ind_AEN,SBM_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% Dist='cosine';
% [SBM_RI_AEE2,~,SBM_t_AEE2,~,SBM_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);
[Adj,Y]=simGenerate(2,n);
Dist='sqeuclidean';%Dist='cosine';
[DCSBM_RI_AEE,DCSBM_RI_AEN,DCSBM_RI_ASE,DCSBM_t_AEE,DCSBM_t_AEN,DCSBM_t_ASE,DCSBM_ind_AEE,DCSBM_ind_AEN,DCSBM_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% Dist='cosine';
% [DCSBM_RI_AEE2,~,DCSBM_t_AEE2,~,DCSBM_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);
[Adj,Y]=simGenerate(3,n);
Dist='sqeuclidean';%Dist='cosine';
[RDPG_RI_AEE,RDPG_RI_AEN,RDPG_RI_ASE,RDPG_t_AEE,RDPG_t_AEN,RDPG_t_ASE,RDPG_ind_AEE,RDPG_ind_AEN,RDPG_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% Dist='cosine';
% [RDPG_RI_AEE2,~,RDPG_t_AEE2,~,RDPG_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);

%% Real Data
% [AN_RI_AEE,AN_RI_AEN,AN_RI_ASE,AN_t_AEE,AN_t_AEE2,AN_t_AEN,AN_t_ASE]=simClusteringReal(Adj,Label);
% 

%%% 1. Corr better
n=5000;K=10;
[Adj,Y]=simGenerate(10,n);
GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(11,n,K);
GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(12,n,K);
GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(15,n,k);
% GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(16,n,k);
% GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(20,n);
GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(21,n,K);
GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(22,n,K);
GraphClusteringEvaluate(Adj,Y)

% Try K choice
kmax=20;
score=zeros(kmax,1);
ari=zeros(kmax,1);
for r=2:kmax
    [~,Y2,~,~,score(r)]=GraphEncoder(Adj,r);
    ari(r)=RandIndex(Y2,Y+1);
end
subplot(1,2,1);
plot(2:kmax,1-score(2:kmax),'r-','LineWidth',2)
title(strcat('1-MeanSS at K=',num2str(K)))
xlim([2,kmax]);
axis('square')
set(gca,'FontSize',15);
subplot(1,2,2);
plot(2:kmax,ari(2:kmax),'b-','LineWidth',2)
title(strcat('ARI at K=',num2str(K)))
xlim([2,kmax]);
axis('square')
set(gca,'FontSize',15);
%

% [Adj,Y]=simGenerate(25,n,k);
% GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(26,n,k);
% GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(30,n);
% GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(31,n,k);
% GraphClusteringEvaluate(Adj,Y)

load('CoraAdj.mat') %AEL / GFN K=2
GraphClusteringEvaluate(AdjOri,YOri)
load('email.mat') %k=42
GraphClusteringEvaluate(AdjOri,YOri)
load('lastfm.mat') %AEK K=17
GraphClusteringEvaluate(AdjOri,YOri)
load('polblogs.mat') %k=2
GraphClusteringEvaluate(AdjOri,YOri)

% [PB_RI_AEE,PB_RI_AEN,PB_RI_ASE,PB_t_AEE,PB_t_AEE2,PB_t_AEN,PB_t_ASE]=simClusteringReal(Adj,Label);

%%% 3. No changce / similar
load('Adjnoun.mat') %AEL / GFN K=2
GraphClusteringEvaluate(Adj,Y)
load('DD244.mat') %AEL / GFN K=2
GraphClusteringEvaluate(Adj,Y)
load('pubmedAdj.mat') %k=3;
GraphClusteringEvaluate(AdjOri,YOri)
load('Wiki_Data.mat')
GraphClusteringEvaluate(GEAdj,Label+1)
GraphClusteringEvaluate(GFAdj,Label+1)
load('graphCElegans.mat')
GraphClusteringEvaluate(Ag,vcols)
load('adjnoun.mat')
GraphClusteringEvaluate(Adj,Label)
load('Gene.mat') %AEL / GFN K=2
GraphClusteringEvaluate(Adj,Y)
load('IIP.mat') %AEL / GFN K=2
% GraphEncoder(Adj,Y,knum); %0
GraphClusteringEvaluate(Adj,Y)
load('KKI.mat')
GraphClusteringEvaluate(AdjOri,YOri)
load('OHSU.mat')
GraphClusteringEvaluate(AdjOri,YOri)
load('Peking.mat')
GraphClusteringEvaluate(AdjOri,YOri)
% Simple Eval
tic
[~,Y1,~,~]=GraphEncoder(Adj,K);
toc
RandIndex(Y,Y1)
tic
[~,Y1,~,~]=GraphEncoder(Adj,[2:10]);
toc
RandIndex(Y,Y1)