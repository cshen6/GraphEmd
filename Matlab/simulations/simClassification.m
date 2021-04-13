function simClassification

%%% Sims
n=2000;
[Adj,Y]=simGenerate(1,n);
[SBM_error_AEE,SBM_error_ASE,SBM_t_AEE,SBM_t_ASE]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(2,n);
[DCSBM_error_AEE,DCSBM_error_ASE,DCSBM_t_AEE,DCSBM_t_ASE]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(3,n);
[RDPG_error_AEE,RDPG_error_ASE,RDPG_t_AEE,RDPG_t_ASE]=GraphEncoderEvaluate(Adj,Y);

%%% Real Data
load('Wiki_Data.mat')
[GE_error_AEE,GE_error_ASE,GE_t_AEE,GE_t_ASE]=GraphEncoderEvaluate(GEAdj,Label);
[GF_error_AEE,GF_error_ASE,GF_t_AEE,GF_t_ASE]=GraphEncoderEvaluate(GFAdj,Label);
[TE_error_AEE,TE_error_ASE,TE_t_AEE,TE_t_ASE]=GraphEncoderEvaluate(TE,Label);
[TF_error_AEE,TF_error_ASE,TF_t_AEE,TF_t_ASE]=GraphEncoderEvaluate(TF,Label);

load('graphCElegans.mat')
[Ac_error_AEE,Ac_error_ASE,Ac_t_AEE,Ac_t_ASE]=GraphEncoderEvaluate(Ac,vcols);
[Ag_error_AEE,Ag_error_ASE,Ag_t_AEE,Ag_t_ASE]=GraphEncoderEvaluate(Ag,vcols);

