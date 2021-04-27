function simClassification

%%% Sims: neuron size 10, 60 vs 40 split. Work for AEE and AEN
n=2000;
[Adj,Y]=simGenerate(1,n);
[SBM_error_AEE,SBM_error_AEE2,SBM_error_AEN,SBM_error_ASE,SBM_t_AEE,SBM_t_AEE2,SBM_t_AEN,SBM_t_ASE]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(2,n);
[DCSBM_error_AEE,DCSBM_error_AEE2,DCSBM_error_AEN,DCSBM_error_ASE,DCSBM_t_AEE,DCSBM_t_AEE2,DCSBM_t_AEN,DCSBM_t_ASE]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(3,n);
[RDPG_error_AEE,RDPG_error_AEE2,RDPG_error_AEN,RDPG_error_ASE,RDPG_t_AEE,RDPG_t_AEE2,RDPG_t_AEN,RDPG_t_ASE]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(4,n);
[RDPG2_error_AEE,RDPG2_error_AEE2,RDPG2_error_AEN,RDPG2_error_ASE,RDPG2_t_AEE,RDPG2_t_AEE2,RDPG2_t_AEN,RDPG2_t_ASE]=GraphEncoderEvaluate(Adj,Y);

%%% AEE and AEN:
load('graphCElegans.mat')
[Ac_error_AEE,Ac_error_AEE2,Ac_error_AEN,Ac_error_ASE,Ac_t_AEE,Ac_t_AEE2,Ac_t_AEN,Ac_t_ASE]=GraphEncoderEvaluate(Ac,vcols);
[Ag_error_AEE,Ag_error_AEE2,Ag_error_AEN,Ag_error_ASE,Ag_t_AEE,Ag_t_AEE2,Ag_t_AEN,Ag_t_ASE]=GraphEncoderEvaluate(Ag,vcols);

load('adjnoun.mat')
[AN_error_AEE,AN_error_AEE2,AN_error_AEN,AN_error_ASE,AN_t_AEE,AN_t_AEE2,AN_t_AEN,AN_t_ASE]=GraphEncoderEvaluate(Adj,Label);

%%% AEN only:
load('polblogs.mat') 
[PB_error_AEE,PB_error_AEE2,PB_error_AEN,PB_error_ASE,PB_t_AEE,PB_t_AEE2,PB_t_AEN,PB_t_ASE]=GraphEncoderEvaluate(Adj,Label);

%%% Distance and Kernel
load('Wiki_Data.mat')
[TE_error_AEE,TE_error_AEE2,TE_error_AEN,TE_error_ASE,TE_t_AEE,TE_t_AEE2,TE_t_AEN,TE_t_ASE]=GraphEncoderEvaluate(TE,Label);
[TF_error_AEE,TF_error_AEE2,TF_error_AEN,TF_error_ASE,TF_t_AEE,TF_t_AEE2,TF_t_AEN,TF_t_ASE]=GraphEncoderEvaluate(TF,Label);

% 
% 
% 
% %%% Pending
% load('Wiki_Data.mat')
% opts = struct('neuron',30,'epoch',100,'training',0.8); % default parameters
% [GE_error_AEE,GE_error_AEE2,GE_error_AEN,GE_error_ASE,GE_t_AEE,GE_t_AEE2,GE_t_AEN,GE_t_ASE]=GraphEncoderEvaluate(GEAdj,Label,opts);
% [GF_error_AEE,GF_error_AEE2,GF_error_AEN,GF_error_ASE,GF_t_AEE,GF_t_AEE2,GF_t_AEN,GF_t_ASE]=GraphEncoderEvaluate(GFAdj,Label,opts);

load('email.mat')
[EU_error_AEE,EU_error_AEE2,EU_error_AEN,EU_error_ASE,EU_t_AEE,EU_t_AEE2,EU_t_AEN,EU_t_ASE]=GraphEncoderEvaluate(Adj,Label);


%%% ARE

n=300;
[Adj,Y]=simGenerate(1,n);
[SBM_error_AEE,SBM_error_AEE2,SBM_error_ASE,SBM_t_AEE,SBM_t_AEE2,SBM_t_ASE,SBM_error_ARE,SBM_t_ARE]=GraphEncoderEvaluate2(Adj,Y);


