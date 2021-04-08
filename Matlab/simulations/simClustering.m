function simClustering

n=2000;
[Adj,Y]=simGenerate(1,n);
[SBM_RI_AEE,SBM_RI_ASE,SBM_t_AEE,SBM_t_ASE,SBM_ind_AEE,SBM_ind_ASE]=simClusteringReal(Adj,Y);
[Adj,Y]=simGenerate(2,n);
[DCSBM_RI_AEE,DCSBM_RI_ASE,DCSBM_t_AEE,DCSBM_t_ASE,DCSBM_ind_AEE,DCSBM_ind_ASE]=simClusteringReal(Adj,Y);
[Adj,Y]=simGenerate(3,n);
[RDPG_RI_AEE,RDPG_RI_ASE,RDPG_t_AEE,RDPG_t_ASE,RDPG_ind_AEE,RDPG_ind_ASE]=simClusteringReal(Adj,Y);

%% Real Data
load('Wiki_Data.mat')
[GE_RI_AEE,GE_RI_ASE,GE_t_AEE,GE_t_ASE,GE_ind_AEE,GE_ind_ASE]=simClusteringReal(GEAdj,Label);
[GF_RI_AEE,GF_RI_ASE,GF_t_AEE,GF_t_ASE,GF_ind_AEE,GF_ind_ASE]=simClusteringReal(GFAdj,Label);
[TE_RI_AEE,TE_RI_ASE,TE_t_AEE,TE_t_ASE,TE_ind_AEE,TE_ind_ASE]=simClusteringReal(TE,Label);
[TF_RI_AEE,TF_RI_ASE,TF_t_AEE,TF_t_ASE,TF_ind_AEE,TF_ind_ASE]=simClusteringReal(TF,Label);

load('graphCElegans.mat')
[Ac_RI_AEE,Ac_RI_ASE,Ac_t_AEE,Ac_t_ASE,Ac_ind_AEE,Ac_ind_ASE]=simClusteringReal(Ac,vcols);
[Ag_RI_AEE,Ag_RI_ASE,Ag_t_AEE,Ag_t_ASE,Ag_ind_AEE,Ag_ind_ASE]=simClusteringReal(Ag,vcols);