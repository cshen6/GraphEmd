function simFusion

% Figure 1 SBM
lim=10;rep=10;
G11_acc_GFN=zeros(lim,rep); G12_acc_GFN=zeros(lim,rep); G13_acc_GFN=zeros(lim,rep); 
G21_acc_GFN=zeros(lim,rep); G22_acc_GFN=zeros(lim,rep);G23_acc_GFN=zeros(lim,rep);G3_acc_GFN=zeros(lim,rep);
% G11_t_GFN=0; G12_t_GFN=0; G13_t_GFN=0; G21_t_GFN=0; G22_t_GFN=0;G23_t_GFN=0;G3_t_GFN=0;rep=30;
opts = struct('Spectral',1,'Laplacian',0,'Adjacency',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',1,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
for i=1:lim
    for r=1:rep
        n=100*i;
        [Dis,Label]=simGenerate(18,n,1);
        indices = crossvalind('Kfold',Label,10);
        opts.indices=indices;
        result=GraphEncoderEvaluate(Dis{1},Label,opts);
        G11_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate(Dis{2},Label,opts);
        G12_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate(Dis{3},Label,opts);
        G13_acc_GFN(i,r)=result{'acc','GFN'};
        X=zeros(n,n,2);
        result=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
        G21_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
        G22_acc_GFN(i,r)=result{'acc','GFN'};
        result=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
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


opts = struct('Spectral',1,'Laplacian',0,'Adjacency',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
opts = struct('Spectral',1,'Laplacian',0,'Adjacency',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
load('Wiki_Data.mat')
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;
WikiTE=GraphEncoderEvaluate(TE,Label,opts);
WikiTF=GraphEncoderEvaluate(TF,Label,opts);
WikiGE=GraphEncoderEvaluate(GE,Label,opts);
WikiGF=GraphEncoderEvaluate(GF,Label,opts);
WikiText=GraphEncoderEvaluate({TE,TF},Label,opts);
WikiGraph=GraphEncoderEvaluate({GE,GF},Label,opts);
WikiEng=GraphEncoderEvaluate({GE,TE},Label,opts);
WikiFre=GraphEncoderEvaluate({GF,TF},Label,opts);
WikiTextEng=GraphEncoderEvaluate({TE,TF,GE},Label,opts);
WikiTextFre=GraphEncoderEvaluate({TE,TF,GF},Label,opts);
WikiGraphEng=GraphEncoderEvaluate({GE,GF,TE},Label,opts);
WikiGraphFre=GraphEncoderEvaluate({GE,GF,TF},Label,opts);
WikiAll=GraphEncoderEvaluate({TE,TF,GE,GF},Label,opts);

result1=simStructure(21)
result2=simStructure(22)
result3=simStructure(23)
result4=simStructure(24)
result5=simStructure(25)
result6=simStructure(26)
result7=simStructure(27)
result8=simStructure(28)
result9=simStructure(29)
result10=simStructure(10)
result11=simStructure(11)

opts = struct('Spectral',0,'Laplacian',0,'Adjacency',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
load('graphCElegans.mat')
indices = crossvalind('Kfold',vcols,10);
opts.indices=indices;
CEAc=GraphEncoderEvaluate(Ac,vcols,opts);
CEAg=GraphEncoderEvaluate(Ag,vcols,opts);
CEAll=GraphEncoderEvaluate({Ac,Ag},vcols,opts);

%%%%%%%%% MSFT

load('anonymized_msft.mat')
indices=crossvalind('Kfold',Y,10);
opts = struct('Spectral',0,'Laplacian',0,'Adjacency',1,'GFN',0,'GCN',0,'GNN',0,'knn',0,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
opts.indices=indices;
Acc1=GraphEncoderEvaluate(G{1},label,opts);
Acc2=GraphEncoderEvaluate(G{6},label,opts);
Acc3=GraphEncoderEvaluate(G{12},label,opts);
Acc4=GraphEncoderEvaluate(G{18},label,opts);
Acc5=GraphEncoderEvaluate(G{24},label,opts);
Acc12=GraphEncoderEvaluate({G{1},G{6}},label,opts);
Acc123=GraphEncoderEvaluate({G{1},G{6},G{12}},label,opts);
Acc1234=GraphEncoderEvaluate({G{1},G{6},G{12},G{18}},label,opts);
Acc12345=GraphEncoderEvaluate({G{1},G{6},G{12},G{18},G{24}},label,opts);