function simClassification2

%%% Sims: neuron size 10, 60 vs 40 split. Work for AEE and AEN
n=1000;
[Adj,Y]=simGenerate(1,n);
[SBM_error_AEE,SBM_error_AEN,SBM_error_ARE,SBM_error_ARN,SBM_error_ASE,SBM_t_AEE,SBM_t_AEN,SBM_t_ARE,SBM_t_ARN,SBM_t_ASE]=GraphEncoderEvaluate2(Adj,Y);
[Adj,Y]=simGenerate(2,n);
[DCSBM_error_AEE,DCSBM_error_AEN,DCSBM_error_ARE,DCSBM_error_ARN,DCSBM_error_ASE,DCSBM_t_AEE,DCSBM_t_AEN,DCSBM_t_ARE,DCSBM_t_ARN,DCSBM_t_ASE]=GraphEncoderEvaluate2(Adj,Y);
[Adj,Y]=simGenerate(3,n);
[RDPG_error_AEE,RDPG_error_AEN,RDPG_error_ARE,RDPG_error_ARN,RDPG_error_ASE,RDPG_t_AEE,RDPG_t_AEN,RDPG_t_ARE,RDPG_t_ARN,RDPG_t_ASE]=GraphEncoderEvaluate2(Adj,Y);
[Adj,Y]=simGenerate(4,n);
[RDPG2_error_AEE,RDPG2_error_AEE2,RDPG2_error_AEN,RDPG2_error_ASE,RDPG2_t_AEE,RDPG2_t_AEE2,RDPG2_t_AEN,RDPG2_t_ASE]=GraphEncoderEvaluate2(Adj,Y);

%%% AEE and AEN:
load('graphCElegans.mat')
[Ac_error_AEE,Ac_error_AEN,Ac_error_ARE,Ac_error_ARN,Ac_error_ASE,Ac_t_AEE,Ac_t_AEN,Ac_t_ARE,Ac_t_ARN,Ac_t_ASE]=GraphEncoderEvaluate2(Ac,vcols);
[Ag_error_AEE,Ag_error_AEN,Ag_error_ARE,Ag_error_ARN,Ag_error_ASE,Ag_t_AEE,Ag_t_AEN,Ag_t_ARE,Ag_t_ARN,Ag_t_ASE]=GraphEncoderEvaluate2(Ag,vcols);

load('adjnoun.mat')
[An_error_AEE,An_error_AEN,An_error_ARE,An_error_ARN,An_error_ASE,An_t_AEE,An_t_AEN,An_t_ARE,An_t_ARN,An_t_ASE]=GraphEncoderEvaluate2(Adj,Label);

%%% AEN only:
load('polblogs.mat') 
[PB_error_AEE,PB_error_AEN,PB_error_ARE,PB_error_ARN,PB_error_ASE,PB_t_AEE,PB_t_AEN,PB_t_ARE,PB_t_ARN,PB_t_ASE]=GraphEncoderEvaluate2(Adj,Label);

%%% Distance and Kernel
load('Wiki_Data.mat')
[TE_error_AEE,TE_error_AEN,TE_error_ARE,TE_error_ARN,TE_error_ASE,TE_t_AEE,TE_t_AEN,TE_t_ARE,TE_t_ARN,TE_t_ASE]=GraphEncoderEvaluate2(TE,Label);
[TF_error_AEE,TF_error_AEN,TF_error_ARE,TF_error_ARN,TF_error_ASE,TF_t_AEE,TF_t_AEN,TF_t_ARE,TF_t_ARN,TF_t_ASE]=GraphEncoderEvaluate2(TF,Label);
[GE_error_AEE,GE_error_AEN,GE_error_ARE,GE_error_ARN,GE_error_ASE,GE_t_AEE,GE_t_AEN,GE_t_ARE,GE_t_ARN,GE_t_ASE]=GraphEncoderEvaluate2(GEAdj,Label);
[GF_error_AEE,GF_error_AEN,GF_error_ARE,GF_error_ARN,GF_error_ASE,GF_t_AEE,GF_t_AEN,GF_t_ARE,GF_t_ARN,GF_t_ASE]=GraphEncoderEvaluate2(GFAdj,Label);

% 
% 
% 
% %%% Pending
% load('Wiki_Data.mat')
% opts = struct('neuron',30,'epoch',100,'training',0.8); % default parameters
% [GE_error_AEE,GE_error_AEE2,GE_error_AEN,GE_error_ASE,GE_t_AEE,GE_t_AEE2,GE_t_AEN,GE_t_ASE]=GraphEncoderEvaluate(GEAdj,Label,opts);
% [GF_error_AEE,GF_error_AEE2,GF_error_AEN,GF_error_ASE,GF_t_AEE,GF_t_AEE2,GF_t_AEN,GF_t_ASE]=GraphEncoderEvaluate(GFAdj,Label,opts);

load('email.mat')
[EU_error_AEE,EU_error_AEE2,EU_error_AEN,EU_error_ASE,EU_t_AEE,EU_t_AEE2,EU_t_AEN,EU_t_ASE]=GraphEncoderEvaluate2(Adj,Label);