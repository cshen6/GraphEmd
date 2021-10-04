function simClustering

%%% Basic Sims
n=2000;
% opts = struct('ASE',1,'LSE',1,'NN',0,'Dist','cosine','maxIter',20,'normalize',0,'deg',0,'dmax',30); % default parameters
k=10;rep=10; tt=2;
SBM=zeros(2,5,tt);
RDPG=zeros(2,5,tt);
DCSBM=zeros(2,5,tt);
for r=1:rep
[Adj,Y]=simGenerate(10,n,k);
% % Edge=Adj2Edge(Adj);
SBM(:,:,1)=SBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(11,n,k);
% Edge=Adj2Edge(Adj);
SBM(:,:,2)=SBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(30,n,k);
% % Edge=Adj2Edge(Adj);
RDPG(:,:,1)=RDPG(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(31,n,k);
% Edge=Adj2Edge(Adj);
RDPG(:,:,2)=RDPG(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% DC-SBM
[Adj,Y]=simGenerate(20,n);
% % Edge=Adj2Edge(Adj);
DCSBM(:,:,1)=DCSBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(21,n,k);
% Edge=Adj2Edge(Adj);
DCSBM(:,:,2)=DCSBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
end

n=2000;k=3;
[Adj,Y]=simGenerate(10,n,k);
simClusteringReal(Adj,Y);

% Dist
opts.LSE=0;opts2.LSE=0;
[Adj,Y]=simGenerate(40,n);
Dist=GraphClusteringEvaluate(Adj,Y);
[Adj,Y]=simGenerate(41,n,k);
Dist2=GraphClusteringEvaluate(Adj,Y);
% Kernel
[Adj,Y]=simGenerate(50,n);
Kern=GraphClusteringEvaluate(Adj,Y);
[Adj,Y]=simGenerate(51,n,k);
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
load('Wiki_Data.mat')
GraphClusteringEvaluate(GEAdj,Label)
[GE_RI_AEE,GE_RI_AEN,GE_RI_ASE,GE_t_AEE,GE_t_AEN,GE_t_ASE,GE_ind_AEE,GE_ind_AEN,GE_ind_ASE]=simClusteringReal(GEAdj,Label);
[GF_RI_AEE,GF_RI_AEN,GF_RI_ASE,GF_t_AEE,GF_t_AEN,GF_t_ASE,GF_ind_AEE,GF_ind_AEN,GF_ind_ASE]=simClusteringReal(GFAdj,Label);
[TE_RI_AEE,TE_RI_AEN,TE_RI_ASE,TE_t_AEE,TE_t_AEN,TE_t_ASE,TE_ind_AEE,TE_ind_AEN,TE_ind_ASE]=simClusteringReal(TE,Label);
[TF_RI_AEE,TF_RI_AEN,TF_RI_ASE,TF_t_AEE,TF_t_AEN,TF_t_ASE,TF_ind_AEE,TF_ind_AEN,TF_ind_ASE]=simClusteringReal(TF,Label);

X=zeros(size(TE,1),size(TE,2),4);
X(:,:,1)=TE;X(:,:,2)=TF;X(:,:,3)=GE;X(:,:,4)=GF;
[ind]=GraphClusteringFusion(X,5);
RandIndex(ind,Label+1)

% Dist='cosine';
% [GE_RI_AEE2,~,GE_t_AEE2,~,GE_ind_AEE2,~]=simClusteringReal(GEAdj,Label,Dist);
% [GF_RI_AEE2,~,GF_t_AEE2,~,GF_ind_AEE2,~]=simClusteringReal(GFAdj,Label,Dist);
% [TE_RI_AEE2,~,TE_t_AEE2,~,TE_ind_AEE2,~]=simClusteringReal(TE,Label,Dist);
% [TF_RI_AEE2,~,TF_t_AEE2,~,TF_ind_AEE2,~]=simClusteringReal(TF,Label,Dist);

load('graphCElegans.mat')
[Ac_RI_AEE,Ac_RI_AEN,Ac_RI_ASE,Ac_t_AEE,Ac_t_AEN,Ac_t_ASE,Ac_ind_AEE,Ac_ind_AEN,Ac_ind_ASE]=simClusteringReal(Ac,vcols);
[Ag_RI_AEE,Ag_RI_AEN,Ag_RI_ASE,Ag_t_AEE,Ag_t_AEN,Ag_t_ASE,Ag_ind_AEE,Ag_ind_AEN,Ag_ind_ASE]=simClusteringReal(Ag,vcols);
% Dist='cosine';
% [Ac_RI_AEE2,~,Ac_t_AEE2,~,Ac_ind_AEE2,~]=simClusteringReal(Ac,vcols,Dist);
% [Ag_RI_AEE2,~,Ag_t_AEE2,~,Ag_ind_AEE2,~]=simClusteringReal(Ag,vcols,Dist);


 load('adjnoun.mat')
 GraphClusteringEvaluate(Adj,Label)
% [AN_RI_AEE,AN_RI_AEN,AN_RI_ASE,AN_t_AEE,AN_t_AEE2,AN_t_AEN,AN_t_ASE]=simClusteringReal(Adj,Label);
% 
% %%% AEN only:
load('polblogs.mat') 
GraphClusteringEvaluate(Adj,Label+1)

load('email.mat')
GraphClusteringEvaluate(Adj,Y)

load('pubmed.mat')
GraphClusteringEvaluate(Adj,Y)

load('Gene.mat') %AEL / GFN K=2
GraphClusteringEvaluate(Adj,Y)
load('CoraAdj.mat') %AEL / GFN K=2
GraphClusteringEvaluate(Adj,Y)
% [PB_RI_AEE,PB_RI_AEN,PB_RI_ASE,PB_t_AEE,PB_t_AEE2,PB_t_AEN,PB_t_ASE]=simClusteringReal(Adj,Label);