function simClassification

%%% Sims: neuron size 10, 60 vs 40 split. GFN is the best.
n=1000;
[Adj,Y]=simGenerate(1,n);
[SBM_Acc,SBM_Time]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(2,n);
[DCSBM_Acc,DCSBM_Time]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(3,n);
[RDPG_Acc,RDPG_Time]=GraphEncoderEvaluate(Adj,Y);
[Adj,Y]=simGenerate(4,n);
[RDPG2_Acc,RDPG2_Time]=GraphEncoderEvaluate(Adj,Y);

%%% Repeated Sims
% Figure 1 SBM
SBM_acc_AEE_NNE=0;DCSBM_acc_AEE_NNE=0; RDPG_acc_AEE_NNE=0; SBM_t_AEE_NNE=0; DCSBM_t_AEE_NNE=0;RDPG_t_AEE_NNE=0;
SBM_acc_AEE_NNC=0;DCSBM_acc_AEE_NNC=0; RDPG_acc_AEE_NNC=0; SBM_t_AEE_NNC=0; DCSBM_t_AEE_NNC=0;RDPG_t_AEE_NNC=0;
SBM_acc_AEE_LDA=0;DCSBM_acc_AEE_LDA=0; RDPG_acc_AEE_LDA=0; SBM_t_AEE_LDA=0; DCSBM_t_AEE_LDA=0;RDPG_t_AEE_LDA=0;
SBM_acc_ASE_NNE=0;DCSBM_acc_ASE_NNE=0; RDPG_acc_ASE_NNE=0; SBM_t_ASE_NNE=0; DCSBM_t_ASE_NNE=0;RDPG_t_ASE_NNE=0;
SBM_acc_ASE_LDA=0;DCSBM_acc_ASE_LDA=0; RDPG_acc_ASE_LDA=0; SBM_t_ASE_LDA=0; DCSBM_t_ASE_LDA=0;RDPG_t_ASE_LDA=0;
rep=100;num=20;
for i=1:num
    for r=1:rep
        n=100*i;
        [Adj,Y]=simGenerate(1,n);
        [SBM_Acc,SBM_Time]=GraphEncoderEvaluate(Adj,Y);
        SBM_acc_AEE_NNE(i,r)=SBM_Acc.acc_AEE_NNE;SBM_acc_AEE_NNC(i,r)=SBM_Acc.acc_AEE_NNC;SBM_acc_AEE_LDA(i,r)=SBM_Acc.acc_AEE_LDA;SBM_acc_ASE_NNE(i,r)=SBM_Acc.acc_ASE_NNE;SBM_acc_ASE_LDA(i,r)=SBM_Acc.acc_ASE_LDA;
        SBM_t_AEE_NNE(i,r)=SBM_Time.t_AEE_NNE;SBM_t_AEE_NNC(i,r)=SBM_Time.t_AEE_NNC;SBM_t_AEE_LDA(i,r)=SBM_Time.t_AEE_LDA;SBM_t_ASE_NNE(i,r)=SBM_Time.t_ASE_NNE;SBM_t_ASE_LDA(i,r)=SBM_Time.t_ASE_LDA;
        [Adj,Y]=simGenerate(2,n);
        [DCSBM_Acc,DCSBM_Time]=GraphEncoderEvaluate(Adj,Y);
        DCSBM_acc_AEE_NNE(i,r)=DCSBM_Acc.acc_AEE_NNE;DCSBM_acc_AEE_NNC(i,r)=DCSBM_Acc.acc_AEE_NNC;DCSBM_acc_AEE_LDA(i,r)=DCSBM_Acc.acc_AEE_LDA;DCSBM_acc_ASE_NNE(i,r)=DCSBM_Acc.acc_ASE_NNE;DCSBM_acc_ASE_LDA(i,r)=DCSBM_Acc.acc_ASE_LDA;
        DCSBM_t_AEE_NNE(i,r)=DCSBM_Time.t_AEE_NNE;DCSBM_t_AEE_NNC(i,r)=DCSBM_Time.t_AEE_NNC;DCSBM_t_AEE_LDA(i,r)=DCSBM_Time.t_AEE_LDA;DCSBM_t_ASE_NNE(i,r)=DCSBM_Time.t_ASE_NNE;DCSBM_t_ASE_LDA(i,r)=DCSBM_Time.t_ASE_LDA;
        [Adj,Y]=simGenerate(3,20*i);
        [RDPG_Acc,RDPG_Time]=GraphEncoderEvaluate(Adj,Y);
        RDPG_acc_AEE_NNE(i,r)=RDPG_Acc.acc_AEE_NNE;RDPG_acc_AEE_NNC(i,r)=RDPG_Acc.acc_AEE_NNC;RDPG_acc_AEE_LDA(i,r)=RDPG_Acc.acc_AEE_LDA;RDPG_acc_ASE_NNE(i,r)=RDPG_Acc.acc_ASE_NNE;RDPG_acc_ASE_LDA(i,r)=RDPG_Acc.acc_ASE_LDA;
        RDPG_t_AEE_NNE(i,r)=RDPG_Time.t_AEE_NNE;RDPG_t_AEE_NNC(i,r)=RDPG_Time.t_AEE_NNC;RDPG_t_AEE_LDA(i,r)=RDPG_Time.t_AEE_LDA;RDPG_t_ASE_NNE(i,r)=RDPG_Time.t_ASE_NNE;RDPG_t_ASE_LDA(i,r)=RDPG_Time.t_ASE_LDA;
    end
end

%%% AEE and AEN:
load('graphCElegans.mat')
[CEAc_Acc,CEAc_Time]=GraphEncoderEvaluate(Ac,vcols);
[CEAg_Acc,CEAg_Time]=GraphEncoderEvaluate(Ag,vcols);

load('adjnoun.mat')
[AN_Acc,AN_Time]=GraphEncoderEvaluate(Adj,Label);

%%% AEN only:
load('polblogs.mat') 
[PB_Acc,PB_Time]=GraphEncoderEvaluate(Adj,Label);

%%% Distance and Kernel
load('Wiki_Data.mat')
[WikiTE_Acc,WikiTE_Time]=GraphEncoderEvaluate(TE,Label);
[WikiTF_Acc,WikiTF_Time]=GraphEncoderEvaluate(TF,Label);
% D=diag(sum(GEAdj,1));
% GEAdj=D^-0.5*(GEAdj+eye(size(GEAdj,1)))*D^-0.5;
[WikiGE_Acc,WikiGE_Time]=GraphEncoderEvaluate(GEAdj,Label);
[WikiGF_Acc,WikiGF_Time]=GraphEncoderEvaluate(GFAdj,Label);
% 
% 
% 
% %%% Pending
% load('Wiki_Data.mat')
% opts = struct('neuron',30,'epoch',100,'training',0.8); % default parameters
% [GE_acc_AEE,GE_acc_AEE2,GE_acc_AEN,GE_acc_ASE,GE_t_AEE,GE_t_AEE2,GE_t_AEN,GE_t_ASE]=GraphEncoderEvaluate(GEAdj,Label,opts);
% [GF_acc_AEE,GF_acc_AEE2,GF_acc_AEN,GF_acc_ASE,GF_t_AEE,GF_t_AEE2,GF_t_AEN,GF_t_ASE]=GraphEncoderEvaluate(GFAdj,Label,opts);

%%% when k is large relative to n, GCN overfit???

load('Cora.mat') %AEK K=7
[Cora_Acc,Cora_Time]=GraphEncoderEvaluate(Adj,Y);

load('DD244.mat') %GCN K=20
[DD_Acc,DD_Time]=GraphEncoderEvaluate(Adj,Y);

load('Gene.mat') %AEL / GFN K=2
[Gene_Acc,Gene_Time]=GraphEncoderEvaluate(Adj,Y);

load('KKI.mat')%GCN K=189
[KKI_Acc,KKI_Time]=GraphEncoderEvaluate(Adj,Y);

load('lastfm.mat') %AEK K=17
[LFM_Acc,LFM_Time]=GraphEncoderEvaluate(Adj,Y);

load('OHSU.mat')%GCN K=189
[OHSU_Acc,OHSU_Time]=GraphEncoderEvaluate(Adj,Y);

load('Peking.mat')%GCN K=189
[Pek_Acc,Pek_Time]=GraphEncoderEvaluate(Adj,Y);

load('pubmed.mat')
[Pub_Acc,Pub_Time]=GraphEncoderEvaluate(Adj,Y);


% %%% GCN False Example
% n=2000;k=5;p=0.4;
% Y=randi(k,n,1);
% X=unifrnd(0,1,n,n);
% X=(X+X')/2;
% X=double(X<p);
% for i=1:n
% X(i,i)=0;
% end
% [acc_AEL,t_AEL, acc_GFN, t_GFN, acc_ASE, t_ASE, acc_GCN,t_GCN, acc_AEK,t_AEK,acc_ANN,t_ANN]=GraphEncoderEvaluate(X,Y);

%%% ARE

n=300;
[Adj,Y]=simGenerate(1,n);
[SBM_acc_AEE,SBM_acc_AEE2,SBM_acc_ASE,SBM_t_AEE,SBM_t_AEE2,SBM_t_ASE,SBM_acc_ARE,SBM_t_ARE]=GraphEncoderEvaluate2(Adj,Y);


