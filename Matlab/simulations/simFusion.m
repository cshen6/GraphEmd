function simFusion

% Figure 1 SBM
G11_acc_AEN=0; G12_acc_AEN=0; G13_acc_AEN=0; G21_acc_AEN=0; G22_acc_AEN=0;G23_acc_AEN=0;G3_acc_AEN=0;
G11_t_AEN=0; G12_t_AEN=0; G13_t_AEN=0; G21_t_AEN=0; G22_t_AEN=0;G23_t_AEN=0;G3_t_AEN=0;rep=30;
for i=1:10
    for r=1:rep
        n=100*i;
        [Dis,Label]=simGenerate(11,n,1);
        indices = crossvalind('Kfold',Label,10);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN, acc_ASE, t_ASE, acc_GCN(i,r),t_GCN, acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(Dis(:,:,1),Label,indices);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN, acc_ASE, t_ASE, acc_GCN(i,r),t_GCN, acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(Dis(:,:,2),Label,indices);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN, acc_ASE, t_ASE, acc_GCN(i,r),t_GCN, acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(Dis(:,:,3),Label,indices);
        X=zeros(n,n,2);
        X(:,:,1)=Dis(:,:,1);X(:,:,2)=Dis(:,:,3);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN(i,r), acc_ASE(i,r), t_ASE(i,r), acc_GCN(i,r),t_GCN(i,r), acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(X,Label,indices);
        X(:,:,1)=Dis(:,:,2);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN(i,r), acc_ASE(i,r), t_ASE(i,r), acc_GCN(i,r),t_GCN(i,r), acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(X,Label,indices);
        X(:,:,2)=Dis(:,:,1);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN(i,r), acc_ASE(i,r), t_ASE(i,r), acc_GCN(i,r),t_GCN(i,r), acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(X,Label,indices);
        [acc_AEL,t_AEL, acc_GFN(i,r), t_GFN(i,r), acc_ASE(i,r), t_ASE(i,r), acc_GCN(i,r),t_GCN(i,r), acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(Dis,Label,indices);
    end
end
G11_acc_AEN=mean(G11_acc_AEN,2);
G12_acc_AEN=mean(G12_acc_AEN,2);
G13_acc_AEN=mean(G13_acc_AEN,2);
G21_acc_AEN=mean(G21_acc_AEN,2);
G22_acc_AEN=mean(G22_acc_AEN,2);
G23_acc_AEN=mean(G23_acc_AEN,2);
G3_acc_AEN=mean(G3_acc_AEN,2);



load('Wiki_Data.mat')
indices = crossvalind('Kfold',Label,10);
[Wiki_TE_acc_AEE,Wiki_TE_acc_AEE2,Wiki_TE_acc_AEN,Wiki_TE_acc_ASE,Wiki_TE_t_AEE,Wiki_TE_t_AEE2,Wiki_TE_t_AEN,Wiki_TE_t_ASE]=GraphEncoderEvaluate(TE,Label,indices);
[Wiki_TF_acc_AEE,Wiki_TF_acc_AEE2,Wiki_TF_acc_AEN,Wiki_TF_acc_ASE,Wiki_TF_t_AEE,Wiki_TF_t_AEE2,Wiki_TF_t_AEN,Wiki_TF_t_ASE]=GraphEncoderEvaluate(TF,Label,indices);
[Wiki_GE_acc_AEE,Wiki_GE_acc_AEE2,Wiki_GE_acc_AEN,Wiki_GE_acc_ASE,Wiki_GE_t_AEE,Wiki_GE_t_AEE2,Wiki_GE_t_AEN,Wiki_GE_t_ASE]=GraphEncoderEvaluate(GE,Label,indices);
[Wiki_GF_acc_AEE,Wiki_GF_acc_AEE2,Wiki_GF_acc_AEN,Wiki_GF_acc_ASE,Wiki_GF_t_AEE,Wiki_GF_t_AEE2,Wiki_GF_t_AEN,Wiki_GF_t_ASE]=GraphEncoderEvaluate(GF,Label,indices);

X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=TE;X(:,:,2)=TF;
[Wiki_Text_acc_AEE,~,Wiki_Text_acc_AEN,Wiki_Text_acc_ASE,Wiki_Text_t_AEE,~,Wiki_Text_t_AEN,Wiki_Text_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=GE;X(:,:,2)=GF;
[Wiki_Graph_acc_AEE,~,Wiki_Graph_acc_AEN,Wiki_Graph_acc_ASE,Wiki_Graph_t_AEE,~,Wiki_Graph_t_AEN,Wiki_Graph_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(TE,1),size(GE,2),2);
X(:,:,1)=TE;X(:,:,2)=GE;
[Wiki_Eng_acc_AEE,~,Wiki_Eng_acc_AEN,Wiki_Text_acc_ASE,Wiki_Text_t_AEE,~,Wiki_Text_t_AEN,Wiki_Text_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(TE,1),size(TE,2),2);
X(:,:,1)=TF;X(:,:,2)=GF;
[Wiki_Fre_acc_AEE,~,Wiki_Fre_acc_AEN,Wiki_Fre_acc_ASE,Wiki_Fre_t_AEE,~,Wiki_Fre_t_AEN,Wiki_Fre_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(TE,1),size(TE,2),4);
X(:,:,1)=TE;X(:,:,2)=TF;X(:,:,3)=GE;X(:,:,4)=GF;
[Wiki_acc_AEE,~,Wiki_acc_AEN,Wiki_acc_ASE,Wiki_t_AEE,~,Wiki_t_AEN,Wiki_t_ASE]=GraphEncoderEvaluate(X,Label);


load('graphCElegans.mat')
indices = crossvalind('Kfold',vcols,10);
[CE_Ac_acc_AEE,CE_Ac_acc_AEE2,CE_Ac_acc_AEN,CE_Ac_acc_ASE,CE_Ac_t_AEE,CE_Ac_t_AEE2,CE_Ac_t_AEN,CE_Ac_t_ASE]=GraphEncoderEvaluate(Ac,vcols,indices);
[CE_Ag_acc_AEE,CE_Ag_acc_AEE2,CE_Ag_acc_AEN,CE_Ag_acc_ASE,CE_Ag_t_AEE,CE_Ag_t_AEE2,CE_Ag_t_AEN,CE_Ag_t_ASE]=GraphEncoderEvaluate(Ag,vcols,indices);
X=zeros(size(Ac,1),size(Ac,2),2);
X(:,:,1)=Ac;X(:,:,2)=Ag;
[CE_acc_AEE,~,CE_acc_AEN,CE_acc_ASE,CE_t_AEE,~,CE_t_AEN,CE_t_ASE]=GraphEncoderEvaluate(X,vcols);


load('pubmed.mat')
load('cora.mat')


load('BrainHippoShape.mat')
[Brain_LML_acc_AEE,Brain_LML_acc_AEE2,Brain_LML_acc_AEN,Brain_LML_acc_ASE,Brain_LML_t_AEE,Brain_LML_t_AEE2,Brain_LML_t_AEN,Brain_LML_t_ASE]=GraphEncoderEvaluate(LML,Label,indices);
[Brain_LMR_acc_AEE,Brain_LMR_acc_AEE2,Brain_LMR_acc_AEN,Brain_LMR_acc_ASE,Brain_LMR_t_AEE,Brain_LMR_t_AEE2,Brain_LMR_t_AEN,Brain_LMR_t_ASE]=GraphEncoderEvaluate(LMR,Label,indices);
[Brain_SML_acc_AEE,Brain_SML_acc_AEE2,Brain_SML_acc_AEN,Brain_SML_acc_ASE,Brain_SML_t_AEE,Brain_SML_t_AEE2,Brain_SML_t_AEN,Brain_SML_t_ASE]=GraphEncoderEvaluate(SML,Label,indices);
[Brain_SMR_acc_AEE,Brain_SMR_acc_AEE2,Brain_SMR_acc_AEN,Brain_SMR_acc_ASE,Brain_SMR_t_AEE,Brain_SMR_t_AEE2,Brain_SMR_t_AEN,Brain_SMR_t_ASE]=GraphEncoderEvaluate(SMR,Label,indices);
X=zeros(size(LML,1),size(LML,2),2);
X(:,:,1)=LML;X(:,:,2)=LMR;
[Brain_L_acc_AEE,Brain_L_acc_AEN,Brain_L_acc_ASE,Brain_L_t_AEE,Brain_L_t_AEN,Brain_L_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(LML,1),size(LML,2),2);
X(:,:,1)=SML;X(:,:,2)=SMR;
[Brain_S_acc_AEE,Brain_S_acc_AEN,Brain_S_acc_ASE,Brain_S_t_AEE,Brain_S_t_AEN,Brain_S_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
X=zeros(size(LML,1),size(LML,2),4);
X(:,:,1)=LML;X(:,:,2)=LMR;X(:,:,3)=SML;X(:,:,4)=SMR;
[Brain_acc_AEE,Brain_acc_AEN,Brain_acc_ASE,Brain_t_AEE,Brain_t_AEN,Brain_t_ASE]=GraphEncoderEvaluate(X,Label,indices);
