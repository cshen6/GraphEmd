function simFusion(choice)

% Figure 1 SBM
if choice==1
    lim=10;rep=1;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
%     G11_acc=zeros(lim,rep); G12_acc=zeros(lim,rep); G13_acc=zeros(lim,rep);
%     G21_acc=zeros(lim,rep); G22_acc=zeros(lim,rep);G23_acc_GFN=zeros(lim,rep);G3_acc_GFN=zeros(lim,rep);
    % G11_t_GFN=0; G12_t_GFN=0; G13_t_GFN=0; G21_t_GFN=0; G22_t_GFN=0;G23_t_GFN=0;G3_t_GFN=0;rep=30;
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',1,'LDA',0,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(18,n,1);
%             Dis={adj2edge(Dis{1}),adj2edge(Dis{2}),adj2edge(Dis{3})};
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
%             X=zeros(n,n,2);
            G13{i,r}=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
            G23{i,r}=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
            G12{i,r}=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
        end
    end
end

if choice==2
    lim=10;rep=1;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
%     G11_acc=zeros(lim,rep); G12_acc=zeros(lim,rep); G13_acc=zeros(lim,rep);
%     G21_acc=zeros(lim,rep); G22_acc=zeros(lim,rep);G23_acc_GFN=zeros(lim,rep);G3_acc_GFN=zeros(lim,rep);
    % G11_t_GFN=0; G12_t_GFN=0; G13_t_GFN=0; G21_t_GFN=0; G22_t_GFN=0;G23_t_GFN=0;G3_t_GFN=0;rep=30;
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',1,'LDA',0,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(28,n,1);
%             Dis={adj2edge(Dis{1}),adj2edge(Dis{2}),adj2edge(Dis{3})};
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
%             X=zeros(n,n,2);
            G13{i,r}=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
            G23{i,r}=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
            G12{i,r}=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
        end
    end
end

if choice==3
    load('Wiki_Data.mat'); Label=Label+1;
    opts = struct('Adjacency',1,'DiagAugment',0,'Laplacian',0,'Spectral',1,'LDA',0,'GNN',1,'knn',5,'dim',30);
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
    % result1=simStructure(21)
    % result2=simStructure(22)
    % result3=simStructure(23)
    % result4=simStructure(24)
    % result5=simStructure(25)
    % result6=simStructure(26)
    % result7=simStructure(27)
    % result8=simStructure(28)
    % result9=simStructure(29)
    % result10=simStructure(10)
    % result11=simStructure(11)
end

if choice==4
    load('graphCElegans.mat')
    opts = struct('Adjacency',1,'DiagAugment',0,'Laplacian',0,'Spectral',1,'LDA',0,'GNN',1,'knn',5,'dim',30);indices = crossvalind('Kfold',vcols,10);
    indices = crossvalind('Kfold',vcols,10);
    opts.indices=indices;
    CEAc=GraphEncoderEvaluate(Ac,vcols,opts);
    CEAg=GraphEncoderEvaluate(Ag,vcols,opts);
    CEAll=GraphEncoderEvaluate({Ac,Ag},vcols,opts);
end
%%%%%%%%% MSFT

if choice==5
    load('anonymized_msft.mat')
    indices=crossvalind('Kfold',Y,10);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',1,'knn',5,'dim',30);
    opts.indices=indices;
    Acc1=GraphEncoderEvaluate(G{1},label,opts);
    Acc2=GraphEncoderEvaluate(G{6},label,opts);
    Acc3=GraphEncoderEvaluate(G{12},label,opts);
    Acc4=GraphEncoderEvaluate(G{18},label,opts);
    Acc5=GraphEncoderEvaluate(G{24},label,opts);
    Acc12=GraphEncoderEvaluate({G{1},G{6}},label,opts);
    Acc123=GraphEncoderEvaluate({G{1},G{6},G{12}},label,opts);
    Acc1234=GraphEncoderEvaluate({G{1},G{6},G{12},G{18}},label,opts);
    tic
    Acc12345=GraphEncoderEvaluate({G{1},G{6},G{12},G{18},G{24}},label,opts);toc
end