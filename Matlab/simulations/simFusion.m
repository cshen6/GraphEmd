function simFusion


load('Wiki_Data.mat')
[Wiki_TE_error_AEE,Wiki_TE_error_AEE2,Wiki_TE_error_AEN,Wiki_TE_error_ASE,Wiki_TE_t_AEE,Wiki_TE_t_AEE2,Wiki_TE_t_AEN,Wiki_TE_t_ASE]=GraphEncoderEvaluate(TE,Label);
[Wiki_TF_error_AEE,Wiki_TF_error_AEE2,Wiki_TF_error_AEN,Wiki_TF_error_ASE,Wiki_TF_t_AEE,Wiki_TF_t_AEE2,Wiki_TF_t_AEN,Wiki_TF_t_ASE]=GraphEncoderEvaluate(TF,Label);
[Wiki_GE_error_AEE,Wiki_GE_error_AEE2,Wiki_GE_error_AEN,Wiki_GE_error_ASE,Wiki_GE_t_AEE,Wiki_GE_t_AEE2,Wiki_GE_t_AEN,Wiki_GE_t_ASE]=GraphEncoderEvaluate(GE,Label);
[Wiki_GF_error_AEE,Wiki_GF_error_AEE2,Wiki_GF_error_AEN,Wiki_GF_error_ASE,Wiki_GF_t_AEE,Wiki_GF_t_AEE2,Wiki_GF_t_AEN,Wiki_GF_t_ASE]=GraphEncoderEvaluate(GF,Label);

X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=TE;X(:,:,2)=TF;
[Wiki_Text_error_AEE,~,Wiki_Text_error_AEN,Wiki_Text_error_ASE,Wiki_Text_t_AEE,~,Wiki_Text_t_AEN,Wiki_Text_t_ASE]=GraphEncoderEvaluate(X,Label);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=GE;X(:,:,2)=GF;
[Wiki_Graph_error_AEE,~,Wiki_Graph_error_AEN,Wiki_Graph_error_ASE,Wiki_Graph_t_AEE,~,Wiki_Graph_t_AEN,Wiki_Graph_t_ASE]=GraphEncoderEvaluate(X,Label);
X=zeros(size(TE,1),size(TE,2),4);
X(:,:,1)=TE;X(:,:,2)=TF;X(:,:,3)=GE;X(:,:,4)=GF;
[Wiki_error_AEE,~,Wiki_error_AEN,Wiki_error_ASE,Wiki_t_AEE,~,Wiki_t_AEN,Wiki_t_ASE]=GraphEncoderEvaluate(X,Label);


load('graphCElegans.mat')
[CE_Ac_error_AEE,CE_Ac_error_AEE2,CE_Ac_error_AEN,CE_Ac_error_ASE,CE_Ac_t_AEE,CE_Ac_t_AEE2,CE_Ac_t_AEN,CE_Ac_t_ASE]=GraphEncoderEvaluate(Ac,vcols);
[CE_Ag_error_AEE,CE_Ag_error_AEE2,CE_Ag_error_AEN,CE_Ag_error_ASE,CE_Ag_t_AEE,CE_Ag_t_AEE2,CE_Ag_t_AEN,CE_Ag_t_ASE]=GraphEncoderEvaluate(Ag,vcols);
X=zeros(size(Ac,1),size(Ac,2),2);
X(:,:,1)=Ac;X(:,:,2)=Ag;
[CE_error_AEE,~,CE_error_AEN,CE_error_ASE,CE_t_AEE,~,CE_t_AEN,CE_t_ASE]=GraphEncoderEvaluate(X,vcols);

% load('BrainHippoShape.mat')
% [Brain_LML_error_AEE,Brain_LML_error_AEE2,Brain_LML_error_AEN,Brain_LML_error_ASE,Brain_LML_t_AEE,Brain_LML_t_AEE2,Brain_LML_t_AEN,Brain_LML_t_ASE]=GraphEncoderEvaluate(LML,Label);
% [Brain_LMR_error_AEE,Brain_LMR_error_AEE2,Brain_LMR_error_AEN,Brain_LMR_error_ASE,Brain_LMR_t_AEE,Brain_LMR_t_AEE2,Brain_LMR_t_AEN,Brain_LMR_t_ASE]=GraphEncoderEvaluate(LMR,Label);
% [Brain_SML_error_AEE,Brain_SML_error_AEE2,Brain_SML_error_AEN,Brain_SML_error_ASE,Brain_SML_t_AEE,Brain_SML_t_AEE2,Brain_SML_t_AEN,Brain_SML_t_ASE]=GraphEncoderEvaluate(SML,Label);
% [Brain_SMR_error_AEE,Brain_SMR_error_AEE2,Brain_SMR_error_AEN,Brain_SMR_error_ASE,Brain_SMR_t_AEE,Brain_SMR_t_AEE2,Brain_SMR_t_AEN,Brain_SMR_t_ASE]=GraphEncoderEvaluate(SMR,Label);
% X=zeros(size(LML,1),size(LML,2),4);
% X(:,:,1)=LML;X(:,:,2)=LMR;X(:,:,3)=SML;X(:,:,4)=SMR;
% [Brain_error_AEE,Brain_error_AEN,Brain_error_ASE,Brain_t_AEE,Brain_t_AEN,Brain_t_ASE]=GraphEncoderFusionEvaluate(X,Label);
