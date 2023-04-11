function simRefine(choice,spec, rep)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<2
    spec=0;
end
if nargin<3
    rep=2;
end
% Figure 1 SBM
if choice==1 || choice==2
    lim=3;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=1000*i
            [Dis,Label]=simGenerate(18+(choice-1)*10,n,1);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
            G13{i,r}=GraphEncoderEvaluate(Dis{1},{Label,Label(randperm(n))},opts);
            G23{i,r}=GraphEncoderEvaluate(Dis,{Label,Label(randperm(n))},opts);
            G12{i,r}=GraphEncoderEvaluate(Dis,Label,opts,Label(randperm(n)));
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
        end
    end
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);Acc12=zeros(lim,6);Acc23=zeros(lim,6);Acc13=zeros(lim,6);Acc123=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,1}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,2}/rep;Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,4}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,1}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,2}/rep;Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,4}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,1}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,2}/rep;Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,4}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,1}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,2}/rep;Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,4}/rep;
            Acc3(i,1)=Acc3(i,1)+G3{i,r}{1,1}/rep;Acc3(i,2)=Acc3(i,2)+G3{i,r}{1,2}/rep;Acc3(i,3)=Acc3(i,3)+G3{i,r}{1,4}/rep;
            Acc3(i,4)=Acc3(i,4)+G3{i,r}{4,1}/rep;Acc3(i,5)=Acc3(i,5)+G3{i,r}{4,2}/rep;Acc3(i,6)=Acc3(i,6)+G3{i,r}{4,4}/rep;
            Acc12(i,1)=Acc12(i,1)+G12{i,r}{1,1}/rep;Acc12(i,2)=Acc12(i,2)+G12{i,r}{1,2}/rep;Acc12(i,3)=Acc12(i,3)+G12{i,r}{1,4}/rep;
            Acc12(i,4)=Acc12(i,4)+G12{i,r}{4,1}/rep;Acc12(i,5)=Acc12(i,5)+G12{i,r}{4,2}/rep;Acc12(i,6)=Acc12(i,6)+G12{i,r}{4,4}/rep;
            Acc13(i,1)=Acc13(i,1)+G13{i,r}{1,1}/rep;Acc13(i,2)=Acc13(i,2)+G13{i,r}{1,2}/rep;Acc13(i,3)=Acc13(i,3)+G13{i,r}{1,4}/rep;
            Acc13(i,4)=Acc13(i,4)+G13{i,r}{4,1}/rep;Acc13(i,5)=Acc13(i,5)+G13{i,r}{4,2}/rep;Acc13(i,6)=Acc13(i,6)+G13{i,r}{4,4}/rep;
            Acc23(i,1)=Acc23(i,1)+G23{i,r}{1,1}/rep;Acc23(i,2)=Acc23(i,2)+G23{i,r}{1,2}/rep;Acc23(i,3)=Acc23(i,3)+G23{i,r}{1,4}/rep;
            Acc23(i,4)=Acc23(i,4)+G23{i,r}{4,1}/rep;Acc23(i,5)=Acc23(i,5)+G23{i,r}{4,2}/rep;Acc23(i,6)=Acc23(i,6)+G23{i,r}{4,4}/rep;
            Acc123(i,1)=Acc123(i,1)+G123{i,r}{1,1}/rep;Acc123(i,2)=Acc123(i,2)+G123{i,r}{1,2}/rep;Acc123(i,3)=Acc123(i,3)+G123{i,r}{1,4}/rep;
            Acc123(i,4)=Acc123(i,4)+G123{i,r}{4,1}/rep;Acc123(i,5)=Acc123(i,5)+G123{i,r}{4,2}/rep;Acc123(i,6)=Acc123(i,6)+G123{i,r}{4,4}/rep;
        end
    end
    save(strcat('GEERefineSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc12','Acc13','Acc23','Acc123');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc12);mean(Acc13);mean(Acc23);mean(Acc123)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc12);std(Acc13);std(Acc23);std(Acc123)]
end

if choice==3
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G4=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(11,n,5);
            Dis1=simGenerate(11,n,5);
            Dis2=simGenerate(11,n,5);
            Dis3=simGenerate(11,n,5);
            Dis4=simGenerate(11,n,5);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis1,Label,opts);
            G3{i,r}=GraphEncoderEvaluate({Dis,Dis1},Label,opts);
            G4{i,r}=GraphEncoderEvaluate({Dis,Dis1,Dis2,Dis3,Dis4},Label,opts);
        end
    end
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);Acc4=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,1}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,2}/rep;Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,4}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,1}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,2}/rep;Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,4}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,1}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,2}/rep;Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,4}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,1}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,2}/rep;Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,4}/rep;
            Acc3(i,1)=Acc3(i,1)+G3{i,r}{1,1}/rep;Acc3(i,2)=Acc3(i,2)+G3{i,r}{1,2}/rep;Acc3(i,3)=Acc3(i,3)+G3{i,r}{1,4}/rep;
            Acc3(i,4)=Acc3(i,4)+G3{i,r}{4,1}/rep;Acc3(i,5)=Acc3(i,5)+G3{i,r}{4,2}/rep;Acc3(i,6)=Acc3(i,6)+G3{i,r}{4,4}/rep;
            Acc4(i,1)=Acc4(i,1)+G4{i,r}{1,1}/rep;Acc4(i,2)=Acc4(i,2)+G4{i,r}{1,2}/rep;Acc4(i,3)=Acc4(i,3)+G4{i,r}{1,4}/rep;
            Acc4(i,4)=Acc4(i,4)+G4{i,r}{4,1}/rep;Acc4(i,5)=Acc4(i,5)+G4{i,r}{4,2}/rep;Acc4(i,6)=Acc4(i,6)+G4{i,r}{4,4}/rep;
        end
    end
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc4)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc4)]
end

if choice==4 || choice==5 || choice==6 
    switch choice
        case 4
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols;
        case 5
            load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0);
        case 6
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);
    Acc1=zeros(rep,3);Acc2=zeros(rep,3);Time1=zeros(rep,3);Time2=zeros(rep,3);
    if spec>0 && choice>5
        G1=edge2adj(G1);G2=edge2adj(G2);
    end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,2};Acc2(i,1)=tmp{1,4};Time1(i,1)=tmp{4,2};Time2(i,1)=tmp{4,4};
        tmp=GraphEncoderEvaluate(G2,Label,opts);Acc1(i,2)=tmp{1,2};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,2};Time2(i,2)=tmp{4,4};
        tmp=GraphEncoderEvaluate({G1,G2},Label,opts);Acc1(i,3)=tmp{1,2};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2};Time2(i,3)=tmp{4,4};
    end
    save(strcat('GEEFusion',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end
%%%%%%%%% MSFT

if choice==7
    load('Letter.mat')
    spec=0;
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);
    Acc1=zeros(rep,18);Acc2=zeros(rep,18);Time1=zeros(rep,18);Time2=zeros(rep,18); spc=6;
    lim=10507;
    ind=(Edge1(:,1)>lim) | (Edge1(:,2)>lim);Edge1=Edge1(~ind,:); Label1=Label1(1:lim);
    ind=(Edge2(:,1)>lim) | (Edge2(:,2)>lim);Edge2=Edge2(~ind,:); Label2=Label2(1:lim);
    ind=(Edge3(:,1)>lim) | (Edge3(:,2)>lim);Edge3=Edge3(~ind,:); Label3=Label3(1:lim);
    if spec>0
        G1=edge2adj(Edge1);G2=edge2adj(Edge2);G3=edge2adj(Edge3);
    else
        G1=Edge1;G2=Edge2;G3=Edge3;
    end
    for i=1:rep
        i
        for j=1:3
            switch j
                case 1 
                    Label=Label1;LabelA=Label2;LabelB=Label3;
                case 2
                    Label=Label2;LabelA=Label1;LabelB=Label3;
                case 3
                    Label=Label3;LabelA=Label1;LabelB=Label2;
            end
            indices = crossvalind('Kfold',Label,5);
            opts.indices=indices;
            tmp=GraphEncoderEvaluate(Edge1,Label,opts); Acc1(i,1+(j-1)*spc)=tmp{1,2};Acc2(i,1+(j-1)*spc)=tmp{1,4};Time1(i,1+(j-1)*spc)=tmp{4,2};Time2(i,1+(j-1)*spc)=tmp{4,4};
            tmp=GraphEncoderEvaluate(Edge2,Label,opts); Acc1(i,2+(j-1)*spc)=tmp{1,2};Acc2(i,2+(j-1)*spc)=tmp{1,4};Time1(i,2+(j-1)*spc)=tmp{4,2};Time2(i,2+(j-1)*spc)=tmp{4,4};
            tmp=GraphEncoderEvaluate(Edge3,Label,opts); Acc1(i,3+(j-1)*spc)=tmp{1,2};Acc2(i,3+(j-1)*spc)=tmp{1,4};Time1(i,3+(j-1)*spc)=tmp{4,2};Time2(i,3+(j-1)*spc)=tmp{4,4};
            tmp=GraphEncoderEvaluate({Edge1,Edge2,Edge3},Label,opts); Acc1(i,4+(j-1)*spc)=tmp{1,2};Acc2(i,4+(j-1)*spc)=tmp{1,4};Time1(i,4+(j-1)*spc)=tmp{4,2};Time2(i,4+(j-1)*spc)=tmp{4,4};
%            tmp=GraphEncoderEvaluate(Edge1,Label,opts,[LabelA,LabelB]); Acc1(i,5+(j-1)*spc)=tmp{1,2};Acc2(i,5+(j-1)*spc)=tmp{1,4};Time1(i,5+(j-1)*spc)=tmp{4,2};Time2(i,5+(j-1)*spc)=tmp{4,4};
%             tmp=GraphEncoderEvaluate(Edge2,Label,opts,[LabelA,LabelB]); Acc1(i,5+(j-1)*spc)=tmp{1,2};Acc2(i,5+(j-1)*spc)=tmp{1,4};Time1(i,5+(j-1)*spc)=tmp{4,2};Time2(i,5+(j-1)*spc)=tmp{4,4};
           tmp=GraphEncoderEvaluate({Edge1,Edge2,Edge3},{Label,LabelA,LabelB},opts); Acc1(i,5+(j-1)*spc)=tmp{1,2};Acc2(i,5+(j-1)*spc)=tmp{1,4};Time1(i,5+(j-1)*spc)=tmp{4,2};Time2(i,6+(j-1)*spc)=tmp{4,4};
           tmp=GraphEncoderEvaluate({Edge1,Edge2,Edge3},Label,opts,[LabelA,LabelB]); Acc1(i,6+(j-1)*spc)=tmp{1,2};Acc2(i,6+(j-1)*spc)=tmp{1,4};Time1(i,6+(j-1)*spc)=tmp{4,2};Time2(i,6+(j-1)*spc)=tmp{4,4};
        end
    end
    save(strcat('GEEFusionLetterSpec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==8
    load('Wiki_Data.mat'); Label=Label+1;
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);
    % opts2=opts;
    % opts2.deg=1;opts2.ASE=0;
    Acc1=zeros(rep,13);Acc2=zeros(rep,13);Time1=zeros(rep,13);Time2=zeros(rep,13);
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(1-TE,Label,opts);Acc1(i,1)=tmp{1,2};Acc2(i,1)=tmp{1,4};Time1(i,1)=tmp{4,2};Time2(i,1)=tmp{4,4};
        tmp=GraphEncoderEvaluate(1-TF,Label,opts);Acc1(i,2)=tmp{1,2};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,2};Time2(i,2)=tmp{4,4};
        tmp=GraphEncoderEvaluate(GE,Label,opts);Acc1(i,3)=tmp{1,2};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2};Time2(i,3)=tmp{4,4};
        tmp=GraphEncoderEvaluate(GF,Label,opts);Acc1(i,4)=tmp{1,2};Acc2(i,4)=tmp{1,4};Time1(i,4)=tmp{4,2};Time2(i,4)=tmp{4,4};
        tmp=GraphEncoderEvaluate({1-TE,1-TF},Label,opts);Acc1(i,5)=tmp{1,2};Acc2(i,5)=tmp{1,4};Time1(i,5)=tmp{4,2};Time2(i,5)=tmp{4,4};
        tmp=GraphEncoderEvaluate({GE,GF},Label,opts);Acc1(i,6)=tmp{1,2};Acc2(i,6)=tmp{1,4};Time1(i,6)=tmp{4,2};Time2(i,6)=tmp{4,4};
        tmp=GraphEncoderEvaluate({GE,1-TE},Label,opts);Acc1(i,7)=tmp{1,2};Acc2(i,7)=tmp{1,4};Time1(i,7)=tmp{4,2};Time2(i,7)=tmp{4,4};
        tmp=GraphEncoderEvaluate({GF,1-TF},Label,opts);Acc1(i,8)=tmp{1,2};Acc2(i,8)=tmp{1,4};Time1(i,8)=tmp{4,2};Time2(i,8)=tmp{4,4};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GE},Label,opts);Acc1(i,9)=tmp{1,2};Acc2(i,9)=tmp{1,4};Time1(i,9)=tmp{4,2};Time2(i,9)=tmp{4,4};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GF},Label,opts);Acc1(i,10)=tmp{1,2};Acc2(i,10)=tmp{1,4};Time1(i,10)=tmp{4,2};Time2(i,10)=tmp{4,4};
        tmp=GraphEncoderEvaluate({GE,GF,1-TE},Label,opts);Acc1(i,11)=tmp{1,2};Acc2(i,11)=tmp{1,4};Time1(i,11)=tmp{4,2};Time2(i,11)=tmp{4,4};
        tmp=GraphEncoderEvaluate({GE,GF,1-TF},Label,opts);Acc1(i,12)=tmp{1,2};Acc2(i,12)=tmp{1,4};Time1(i,12)=tmp{4,2};Time2(i,12)=tmp{4,4};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GE,GF},Label,opts);Acc1(i,13)=tmp{1,2};Acc2(i,13)=tmp{1,4};Time1(i,13)=tmp{4,2};Time2(i,13)=tmp{4,4};
    end
    save(strcat('GEEFusionWikiSpec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
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

% if choice==0
%     load('anonymized_msft.mat')
%     indices=crossvalind('Kfold',Y,10);
%     opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
%     opts.indices=indices;
%     Acc1=GraphEncoderEvaluate(G{1},label,opts);
%     Acc2=GraphEncoderEvaluate(G{6},label,opts);
%     Acc3=GraphEncoderEvaluate(G{12},label,opts);
%     Acc4=GraphEncoderEvaluate(G{18},label,opts);
%     Acc5=GraphEncoderEvaluate(G{24},label,opts);
%     Acc12=GraphEncoderEvaluate({G{1},G{6}},label,opts);
%     Acc123=GraphEncoderEvaluate({G{1},G{6},G{12}},label,opts);
%     Acc1234=GraphEncoderEvaluate({G{1},G{6},G{12},G{18}},label,opts);
%     tic
%     Acc12345=GraphEncoderEvaluate({G{1},G{6},G{12},G{18},G{24}},label,opts);toc
%     save(strcat('GEEFusion',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4','Acc5','Acc12','Acc123','Acc1234','Acc12345');
% end

if choice>=10 && choice<=13
    GNN=0;
    switch choice
        case 10
           load('Cora.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));
        case 11
            load('citeseer.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));
        case 12
            load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%spec=0;
        case 13
            load('COIL-RAG.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));%spec=0;
%         case 14
%             load('COX2.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));
%         case 15
%             load('DHFR.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));
%         case 16
%             load('BZR.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));\
%         case 12
%             load('AIDS.mat');K2=30;Dist2='sqeuclidean';Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',GNN,'knn',5,'dim',30);
    Acc1=zeros(rep,8);Acc2=zeros(rep,8);Time1=zeros(rep,8);Time2=zeros(rep,8);
    if spec>0
        Edge=edge2adj(Edge);
    end
    thres=0.2;
    numC=zeros(size(X,2),1);
    LOne=onehotencode(categorical(Label),2);
    for i=1:size(X,2);
        numC(i)=max(abs(corr(X(:,i),LOne)));
    end
    dim=(abs(numC)>thres);
%     Y2=kmeans(X,K2,'Distance',Dist2);
%     tmpZ=GraphEncoder(Edge,0,X);
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices;
       tmp=GraphEncoderEvaluate(Edge,Label,opts); Acc1(i,1)=tmp{1,2-GNN};Acc2(i,1)=tmp{1,4};Time1(i,1)=tmp{4,2-GNN};Time2(i,1)=tmp{4,4};
       tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,2)=tmp{1,2-GNN};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,2-GNN};Time2(i,2)=tmp{4,4};
       tmp=GraphEncoderEvaluate({Edge,D},Label,opts); Acc1(i,3)=tmp{1,2-GNN};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2-GNN};Time2(i,3)=tmp{4,4};
       if spec==0
           tmp=AttributeEvaluate(X,Label,indices); Acc1(i,5)=tmp(1);Time1(i,5)=tmp(2);
%            tmp=AttributeEvaluate(tmpZ,Label,indices); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
           tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Time1(i,7)=tmp{4,2};
           tmp=GraphEncoderEvaluate(Edge,Label,opts,X(:,dim)); Acc1(i,8)=tmp{1,2-GNN};Time1(i,8)=tmp{4,2};
       end
%        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,2-GNN};Acc2(i,4)=tmp{1,4};
%        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,2-GNN};Acc2(i,5)=tmp{1,4};
%        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
%        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEEFusion',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==100 || choice==101;%plot simulations
%         Spec=2;
        i=10;ind=2;lw=4;F.fname='FigFusion1';str1='Classification Error';loc='East';
        if choice==101
            ind=3;F.fname=strcat('FigFusion1Spec',num2str(spec));loc='SouthWest';
        end
        myColor = brewermap(11,'RdYlGn'); myColor2 = brewermap(4,'RdYlBu');myColor(10,:)=myColor2(4,:);
%         myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
        fs=28;
        tl = tiledlayout(1,3);
        nexttile(tl)
        load(strcat('GEEFusionSim1Spec',num2str(spec),'.mat'));
        plot(1:i,Acc1(:,ind),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',lw);hold on
        plot(1:i,Acc2(:,ind),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc3(:,ind),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc12(:,ind),'Color', myColor(7,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc23(:,ind),'Color', myColor(8,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc12(:,ind),'Color', myColor(9,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc123(:,ind),'Color', myColor(10,:), 'LineStyle', '-','LineWidth',lw);
        hold off
        axis('square'); xlim([1,10]);ylim([0,0.8]);yticks([0 0.4 0.8]); xticks([1 5 10]);xticklabels({'100','500','1000'});title('Three SBMs');
        legend('G1','G2', 'G3','G1+G2','G2+G3','G1+G2','G1+G2+G3','Location','SouthWest');
        set(gca,'FontSize',fs); 
        nexttile(tl)
        load(strcat('GEEFusionSim2Spec',num2str(spec),'.mat'));
        plot(1:i,Acc1(:,ind),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',lw);hold on
        plot(1:i,Acc2(:,ind),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc3(:,ind),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc12(:,ind),'Color', myColor(7,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc23(:,ind),'Color', myColor(8,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc12(:,ind),'Color', myColor(9,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc123(:,ind),'Color', myColor(10,:), 'LineStyle', '-','LineWidth',lw);
        hold off
        axis('square'); xlim([1,10]);ylim([0,0.8]);yticks([0 0.4 0.8]); xticks([1 5 10]);xticklabels({'100','500','1000'});title('Three DC-SBMs');
        legend('G1','G2', 'G3','G1+G2','G2+G3','G1+G2','G1+G2+G3','Location','SouthWest');
        set(gca,'FontSize',fs); 
        nexttile(tl)
        load(strcat('GEEFusionSim3Spec',num2str(spec),'.mat'));
        plot(1:i,Acc1(:,ind),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',lw);hold on
        plot(1:i,Acc2(:,ind),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc3(:,ind),'Color', myColor(8,:), 'LineStyle', '-.','LineWidth',lw);
        plot(1:i,Acc4(:,ind),'Color', myColor(10,:), 'LineStyle', '--','LineWidth',lw);
        hold off
        axis('square'); xlim([1,10]);ylim([0,1]);yticks([0 0.5 1]); xticks([1 5 10]);xticklabels({'100','500','1000'});title('Signal + Noise SBM');
        set(gca,'FontSize',fs); 
        ylabel(tl,str1,'FontSize',fs)
        xlabel(tl,'Number of Vertices','FontSize',fs)
        legend('Signal Graph','Noise Graph', 'Signal + 1 Noise','Signal + 5 Noise','Location',loc);
        %         set(gca,'FontSize',fs);

        F.wh=[12 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
end

% if choice==102
%     %%% visualization figure fig4
%     myColor = brewermap(11,'RdYlGn'); myColor2 = brewermap(4,'RdYlBu');myColor(10,:)=myColor2(4,:);
%     fs=30;
%     figure('units','normalized','Position',[0 0 1 1]);
%     [Dis,Label]=simGenerate(18,2000,1);
%     ind1=find(Label==1);
%     ind2=find(Label==2);
%     ind3=find(Label==3);
%     ind4=find(Label==4);
%     ind=[ind1;ind2;ind3;ind4];
%     Label=Label(ind);
%     tl = tiledlayout(2,3);
%     nexttile(tl)
%     imagesc(Dis{1}(ind,ind));
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     colorbar( 'off' )
% %     caxis([-0.2 1]);
%     title('Graph 1 from SBM(B1)')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
%     nexttile(tl)
%     imagesc(Dis{2}(ind,ind));
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     colorbar( 'off' )
%     title('Graph 2 from SBM(B2)')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
%     nexttile(tl)
%     imagesc(Dis{3}(ind,ind));
% %     heatmap(Dis{3}(ind,ind),'GridVisible','off');
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     colorbar( 'off' )
%     title('Graph 3 from SBM(B3)')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
% 
%     Z1=GraphEncoder(Dis{1},Label); 
%     Z2=GraphEncoder({Dis{1},Dis{2}},Label); Z2=horzcat(Z2{:}); 
%     Z3=GraphEncoder(Dis,Label); Z3=horzcat(Z3{:}); 
%     [~,X1]=pca(Z1,'numComponents',3);
%     [~,X2]=pca(Z2,'numComponents',3);
%     [~,X3]=pca(Z3,'numComponents',3);
%     nexttile(tl)
%     hold on
%     plot3(X1(ind1,1),X1(ind1,2),X1(ind1,2),'o','Color', myColor(1,:));
%     plot3(X1(ind2,1),X1(ind2,2),X1(ind2,3),'x','Color', myColor(3,:));
%     plot3(X1(ind3,1),X1(ind3,2),X1(ind3,3),'+','Color', myColor(7,:));
%     plot3(X1(ind4,1),X1(ind4,2),X1(ind4,3),'*','Color', myColor(10,:));
%     hold off
% %     imagesc(Dis{1}(ind,ind));
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     title('Graph 1 Embedding')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
%     nexttile(tl)
%     hold on
%     plot(X2(ind1,1),X2(ind1,2),'o','Color', myColor(1,:),);
%     plot(X2(ind2,1),X2(ind2,2),'Color', myColor(3,:));
%     plot(X2(ind3,1),X2(ind3,2),'Color', myColor(7,:));
%     plot(X2(ind4,1),X2(ind4,2),'Color', myColor(10,:));
%     hold off
% %     imagesc(Dis{1}(ind,ind));
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     title('Graph 1 + 2 Fusion')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
%     nexttile(tl)
%     hold on
%     plot(X3(ind1,1),X3(ind1,2),'Color', myColor(1,:));
%     plot(X3(ind2,1),X3(ind2,2),'Color', myColor(3,:));
%     plot(X3(ind3,1),X3(ind3,2),'Color', myColor(7,:));
%     plot(X3(ind4,1),X3(ind4,2),'Color', myColor(10,:));
%     hold off
% %     imagesc(Dis{1}(ind,ind));
% %     Ax = gca;
% %     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% %     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     title('All Graphs Fusion')
%     set(gca,'FontSize',fs);
%     axis off;
%     axis('square');
% 
%     F.fname='FigFusion0';
%     F.wh=[12 8]*2;
%         %     F.PaperPositionMode='auto';
%         print_fig(gcf,F)
% end