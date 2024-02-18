function simFusion(choice,spec, rep)
% use choice =1 to 3 to replicate the simulation, 11-18 for real data experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<2
    spec=0;
end
if nargin<3
    rep=20;
end
ind=2;ind2=3;ind3=1;%2: 5NN, 3: LDA, 1: two-layer NN.
gcn=0;

% Figure 1 SBM
if choice==1 || choice==2
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',1,'GNN',1,'knn',5,'dim',30,'GCN',gcn);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(18+(choice-1)*10,n,1);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
            G13{i,r}=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
            G23{i,r}=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
            G12{i,r}=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
        end
    end
    Acc1=result_acc(G1);Acc2=result_acc(G2);Acc3=result_acc(G3);
    Acc12=result_acc(G12);Acc23=result_acc(G23);Acc13=result_acc(G13);
    Acc123=result_acc(G123);
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc12','Acc13','Acc23','Acc123');
    [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc12);mean(Acc13);mean(Acc23);mean(Acc123)]
    [std(Acc1);std(Acc2);std(Acc3);std(Acc12);std(Acc13);std(Acc23);std(Acc123)]
end

if choice==3
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G4=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',1,'GNN',1,'knn',5,'dim',30,'GCN',0);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(11,n,5);
            Dis1=simGenerate(11,n,5);
%             Dis2=simGenerate(11,n,5);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis1,Label,opts);
            DisFull={Dis,Dis1};
%             for j=2:5;
%                 DisFull{j}=simGenerate(11,n,5);
%             end
            G3{i,r}=GraphEncoderEvaluate(DisFull,Label,opts);
            DisFull2=cell(1,6);DisFull2(1:2)=DisFull;
            for j=3:6
                DisFull2{j}=simGenerate(11,n,5);
            end
            G4{i,r}=GraphEncoderEvaluate(DisFull2,Label,opts);
        end
    end
    Acc1=result_acc(G1);Acc2=result_acc(G2);Acc3=result_acc(G3);Acc4=result_acc(G4);
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc4)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc4)]
end

% Figure 1 SBM heterogeouns
if choice==4 || choice==5
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',1,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(18+(choice-4)*10,n,1);Dis{1}=Dis{1}*unifrnd(0,0.5);Dis{2}=Dis{2}*unifrnd(0,0.5);Dis{3}=Dis{3}*unifrnd(0,0.5);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
            G13{i,r}=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
            G23{i,r}=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
            G12{i,r}=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            DisFull=cell(1,10);DisFull(1:3)=Dis;
            for j=4:20;
                DisFull{j}=simGenerate(11,n,5)*unifrnd(0.5,2);
            end
            G4{i,r}=GraphEncoderEvaluate(DisFull,Label,opts);
        end
    end
    Acc1=result_acc(G1);Acc2=result_acc(G2);Acc3=result_acc(G3);
    Acc12=result_acc(G12);Acc23=result_acc(G23);Acc13=result_acc(G13);
    Acc123=result_acc(G123);Acc4=result_acc(G4);
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc12','Acc13','Acc23','Acc123','Acc4');
    [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc12);mean(Acc13);mean(Acc23);mean(Acc123)]
    [std(Acc1);std(Acc2);std(Acc3);std(Acc12);std(Acc13);std(Acc23);std(Acc123)]
end

if choice ==6 %time
    lim=10;time1=zeros(lim,rep);time2=zeros(lim,rep);time3=zeros(lim,rep);time4=zeros(lim,rep);
    for i=1:lim
        n=3000*i
        [Dis,Label]=simGenerate(28,n,1);
        A=horzcat(Dis{:});
        for r=1:rep
            tic
            GraphEncoder(Dis{1},Label);
            time1(i,r)=toc;
            tic
            GraphEncoder(Dis,Label);
            time2(i,r)=toc;
            tic
            svds(Dis{1},10);
            time3(i,r)=toc;
            tic
            svds(A,10);
            time4(i,r)=toc;
        end
    end
    [mean(time1,2),mean(time2,2),mean(time3,2),mean(time4,2)]
    [std(time1,[],2),std(time2,[],2),std(time3,[],2),std(time4,[],2)]
    save(strcat('GEEFusionSim',num2str(choice),'.mat'),'lim','n','time1','time2','time3','time4');
end

if choice ==9 %clustering
    lim=4;ARI1=zeros(lim,rep);ARI2=zeros(lim,rep);ARI3=zeros(lim,rep);ARI4=zeros(lim,rep);ARI5=zeros(lim,rep);
    for i=1:lim
        for r=1:rep
            n=1000*i
            [Dis,Label]=simGenerate(28,n,1);
            [G1,O1]=GraphEncoder(Dis{1}*unifrnd(0,0.5),4);
            [G2,O2]=GraphEncoder(Dis{2}*unifrnd(0,0.5),4);
            [G3,O3]=GraphEncoder(Dis{3}*unifrnd(0,0.5),4);
            Z=[G1,G2,G3];
            O4=kmeans(Z,4);
            for j=4:20
                DisNew=simGenerate(11,n,5)*unifrnd(0.5,2);
                tmp=GraphEncoder(DisNew,4);
                Z=[Z,tmp];
            end
            O5=kmeans(Z,4);
            ARI1(i,r)=RandIndex(O1.Y,Label); %RandIndex(O2.Y,Label),RandIndex(O3.Y,Label),RandIndex(O4,Label),RandIndex(O5,Label)];
            ARI2(i,r)=RandIndex(O2.Y,Label);
            ARI3(i,r)=RandIndex(O3.Y,Label);
            ARI4(i,r)=RandIndex(O4,Label);
            ARI5(i,r)=RandIndex(O5,Label);
        end
    end
    [mean(ARI1,2),mean(ARI2,2),mean(ARI3,2),mean(ARI4,2),mean(ARI5,2)]
    [std(ARI1,[],2),std(ARI2,[],2),std(ARI3,[],2),std(ARI4,[],2),std(ARI5,[],2)]
    save(strcat('GEEFusionSim',num2str(choice),'.mat'),'lim','n','ARI1','ARI2','ARI3','ARI4','ARI5');
end

if choice==11 || choice==12 || choice==13 
    switch choice
        case 11
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols;
%         case 12
%             load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0);
        case 13
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
    end
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',1,'GNN',1,'knn',5,'dim',30);
    Acc1=zeros(rep,3);Acc2=zeros(rep,3);Time1=zeros(rep,3);Time2=zeros(rep,3);Acc3=zeros(rep,3);Time3=zeros(rep,3);
    if spec>0 && choice>5
        G1=edge2adj(G1);G2=edge2adj(G2);
    end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc2(i,1)=tmp{1,ind2};Acc3(i,1)=tmp{1,ind3};Time1(i,1)=tmp{4,ind};Time2(i,1)=tmp{4,ind2};Time3(i,1)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(G2,Label,opts);Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({G1,G2},Label,opts);Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
    end
    save(strcat('GEEFusion',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
    [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end
%%%%%%%%% MSFT

if choice==14
    load('Letter.mat')
    spec=0;
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',1,'knn',5,'dim',30);
    Acc1=zeros(rep,12);Acc2=zeros(rep,12);Acc3=zeros(rep,12);Time1=zeros(rep,12);Time2=zeros(rep,12); Time3=zeros(rep,12); spc=4;
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
                    Label=Label1;%LabelA=Label2;LabelB=Label3;
                case 2
                    Label=Label2;%LabelA=Label1;LabelB=Label3;
                case 3
                    Label=Label3;%LabelA=Label1;LabelB=Label2;
            end
                lim=10506;
%             lim=length(Label);
            Label=Label(1:lim);
            indTmp=(Edge1(:,1)>lim) | (Edge1(:,2)>lim);Edge1=Edge1(~indTmp,:); %Label1=Label1(1:lim);
            indTmp=(Edge2(:,1)>lim) | (Edge2(:,2)>lim);Edge2=Edge2(~indTmp,:); %Label2=Label2(1:lim);
            indTmp=(Edge3(:,1)>lim) | (Edge3(:,2)>lim);Edge3=Edge3(~indTmp,:); %Label3=Label3(1:lim);
            indices = crossvalind('Kfold',Label,5);
            opts.indices=indices;
            tmp=GraphEncoderEvaluate(Edge1,Label,opts); Acc1(i,1+(j-1)*spc)=tmp{1,ind};Acc2(i,1+(j-1)*spc)=tmp{1,ind2};Acc3(i,1+(j-1)*spc)=tmp{1,ind3};Time1(i,1+(j-1)*spc)=tmp{4,ind};Time2(i,1+(j-1)*spc)=tmp{4,ind2};Time3(i,1+(j-1)*spc)=tmp{4,ind3};
            tmp=GraphEncoderEvaluate(Edge2,Label,opts); Acc1(i,2+(j-1)*spc)=tmp{1,ind};Acc2(i,2+(j-1)*spc)=tmp{1,ind2};Acc3(i,2+(j-1)*spc)=tmp{1,ind3};Time1(i,2+(j-1)*spc)=tmp{4,ind};Time2(i,2+(j-1)*spc)=tmp{4,ind2};Time3(i,2+(j-1)*spc)=tmp{4,ind3};
            tmp=GraphEncoderEvaluate(Edge3,Label,opts); Acc1(i,3+(j-1)*spc)=tmp{1,ind};Acc2(i,3+(j-1)*spc)=tmp{1,ind2};Acc3(i,3+(j-1)*spc)=tmp{1,ind3};Time1(i,3+(j-1)*spc)=tmp{4,ind};Time2(i,3+(j-1)*spc)=tmp{4,ind2};Time3(i,3+(j-1)*spc)=tmp{4,ind3};
            tmp=GraphEncoderEvaluate({Edge1,Edge2,Edge3},Label,opts); Acc1(i,4+(j-1)*spc)=tmp{1,ind};Acc2(i,4+(j-1)*spc)=tmp{1,ind2};Acc3(i,4+(j-1)*spc)=tmp{1,ind3};Time1(i,4+(j-1)*spc)=tmp{4,ind};Time2(i,4+(j-1)*spc)=tmp{4,ind2};Time3(i,4+(j-1)*spc)=tmp{4,ind3};
            %              tmp=GraphEncoderEvaluate(Edge1,Label,opts,[LabelA,LabelB]); Acc1(i,5+(j-1)*spc)=tmp{1,2};Acc2(i,5+(j-1)*spc)=tmp{1,4};Time1(i,5+(j-1)*spc)=tmp{4,2};Time2(i,5+(j-1)*spc)=tmp{4,4};
            %              tmp=GraphEncoderEvaluate(Edge2,Label,opts,[LabelA,LabelB]); Acc1(i,5+(j-1)*spc)=tmp{1,2};Acc2(i,5+(j-1)*spc)=tmp{1,4};Time1(i,5+(j-1)*spc)=tmp{4,2};Time2(i,5+(j-1)*spc)=tmp{4,4};
            %             tmp=GraphEncoderEvaluate({Edge1,Edge2,Edge3},{Label,LabelA,LabelB},opts); Acc1(i,6+(j-1)*spc)=tmp{1,2};Acc2(i,6+(j-1)*spc)=tmp{1,4};Time1(i,6+(j-1)*spc)=tmp{4,2};Time2(i,6+(j-1)*spc)=tmp{4,4};
        end
    end
    save(strcat('GEEFusionLetterSpec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==15
    load('Wiki_Data.mat'); 
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',1,'knn',5,'dim',30);
    % opts2=opts;
    % opts2.deg=1;opts2.ASE=0;
    Acc1=zeros(rep,13);Acc2=zeros(rep,13);Acc3=zeros(rep,13);Time1=zeros(rep,13);Time2=zeros(rep,13);Time3=zeros(rep,13);
    ind=1;ind3=2;ind2=3;%1: weighted LDA, 2: NN, 3: LDA. 4.ASE*NN, 5. ASE*LDA 
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(1-TE,Label,opts);Acc1(i,1)=tmp{1,ind};Acc2(i,1)=tmp{1,ind2};Acc3(i,1)=tmp{1,ind3};Time1(i,1)=tmp{4,2};Time2(i,1)=tmp{4,ind2};Time3(i,1)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(1-TF,Label,opts);Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(GE,Label,opts);Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(GF,Label,opts);Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Acc3(i,4)=tmp{1,ind3};Time1(i,4)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};Time3(i,4)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({1-TE,1-TF},Label,opts);Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Acc3(i,5)=tmp{1,ind3};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};Time3(i,5)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({GE,GF},Label,opts);Acc1(i,6)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Acc3(i,6)=tmp{1,ind3};Time1(i,6)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};Time3(i,6)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({GE,1-TE},Label,opts);Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,ind2};Acc3(i,7)=tmp{1,ind3};Time1(i,7)=tmp{4,ind};Time2(i,7)=tmp{4,ind2};Time3(i,7)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({GF,1-TF},Label,opts);Acc1(i,8)=tmp{1,ind};Acc2(i,8)=tmp{1,ind2};Acc3(i,8)=tmp{1,ind3};Time1(i,8)=tmp{4,ind};Time2(i,8)=tmp{4,ind2};Time3(i,8)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GE},Label,opts);Acc1(i,9)=tmp{1,ind};Acc2(i,9)=tmp{1,ind2};Acc3(i,9)=tmp{1,ind3};Time1(i,9)=tmp{4,ind};Time2(i,9)=tmp{4,ind2};Time3(i,9)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GF},Label,opts);Acc1(i,10)=tmp{1,ind};Acc2(i,10)=tmp{1,ind2};Acc3(i,10)=tmp{1,ind3};Time1(i,10)=tmp{4,ind};Time2(i,10)=tmp{4,ind2};Time3(i,10)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({GE,GF,1-TE},Label,opts);Acc1(i,11)=tmp{1,ind};Acc2(i,11)=tmp{1,ind2};Acc3(i,11)=tmp{1,ind3};Time1(i,11)=tmp{4,ind};Time2(i,11)=tmp{4,ind2};Time3(i,11)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({GE,GF,1-TF},Label,opts);Acc1(i,12)=tmp{1,ind};Acc2(i,12)=tmp{1,ind2};Acc3(i,12)=tmp{1,ind3};Time1(i,12)=tmp{4,ind};Time2(i,12)=tmp{4,ind2};Time3(i,12)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate({1-TE,1-TF,GE,GF},Label,opts);Acc1(i,13)=tmp{1,ind};Acc2(i,13)=tmp{1,ind2};Acc3(i,13)=tmp{1,ind3};Time1(i,13)=tmp{4,ind};Time2(i,13)=tmp{4,ind2};Time3(i,13)=tmp{4,ind3};
    end
    save(strcat('GEEFusionWikiSpec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
     [std(Acc1);std(Acc2);std(Acc3);std(Time1);std(Time2);std(Time3)]
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

if choice>=16 && choice<=18
    switch choice
        case 16
           load('Cora.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));
        case 17
            load('citeseer.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));
%         case 12
%             load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%spec=0;
        case 18
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
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',1,'knn',5,'dim',30);
    Acc1=zeros(rep,3);Acc2=zeros(rep,3);Time1=zeros(rep,3);Time2=zeros(rep,3);Acc3=zeros(rep,3);Time3=zeros(rep,3);
    if spec>0
        Edge=edge2adj(Edge);
    end
%     thres=0.2;
%     numC=zeros(size(X,2),1);
%     LOne=onehotencode(categorical(Label),2);
%     for i=1:size(X,2);
%         numC(i)=max(abs(corr(X(:,i),LOne)));
%     end
%     dim=(abs(numC)>thres);
%     Y2=kmeans(X,K2,'Distance',Dist2);
%     tmpZ=GraphEncoder(Edge,0,X);
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices;
       tmp=GraphEncoderEvaluate(Edge,Label,opts); Acc1(i,1)=tmp{1,ind};Acc2(i,1)=tmp{1,ind2};Acc3(i,1)=tmp{1,ind3};Time1(i,1)=tmp{4,ind};Time2(i,1)=tmp{4,ind2};Time3(i,1)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate({Edge,D},Label,opts); Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
%        tmp=AttributeEvaluate(X,Label,indices); Acc1(i,4)=tmp(1);Time1(i,4)=tmp(2);
%        if spec==0
% %            tmp=AttributeEvaluate(tmpZ,Label,indices); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Time1(i,7)=tmp{4,2};
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X(:,dim)); Acc1(i,8)=tmp{1,ind};Time1(i,8)=tmp{4,2};
%        end
%        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};
%        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};
%        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
%        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEEFusion',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==100 || choice==101;%plot simulations
         spec=3;
        i=10;ind=3;lw=4;F.fname=strcat('FigFusion1Ind',num2str(ind));str1='Classification Error';loc='East';
        if choice==100
        switch ind
            case 1
                str1='Neural Net Classifier';
            case 2 
                str1='Nearest-Neighbor Classifier';
            case 3
                str1='Linear Discriminant Analysis';
            case 5
                str1='Concatenated GCN';
        end
        end
        if choice==101
            ind=3;F.fname=strcat('FigFusion1Spec',num2str(spec));loc='SouthWest';
            switch spec
            case 1
                str1='Omnibus';
            case 2 
                str1='USE';
            case 3
                str1='MASE';
            end
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

if choice==102;%plot simulations
%         Spec=2;
        i=10;ind=1;lw=4;F.fname=strcat('FigFusion2Ind',num2str(ind));str1='Classification Error';loc='East';
%         switch ind
%             case 1
%                 str1='Neural Net Classifier';
%             case 2 
%                 str1='Nearest-Neighbor Classifier';
%             case 3
%                 str1='Linear Discriminant Analysis';
%             case 5
%                 str1='Concatenated GCN';
%         end
        myColor = brewermap(11,'RdYlGn'); myColor2 = brewermap(4,'RdYlBu');myColor(10,:)=myColor2(4,:);
%         myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
        fs=28;
        tl = tiledlayout(1,2);
        nexttile(tl)
        load('GEEFusionSim4Spec0.mat');
        plot(1:i,Acc1(:,ind),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',lw);hold on
        plot(1:i,Acc2(:,ind),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc3(:,ind),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc123(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',lw);
        plot(1:i,Acc4(:,ind),'Color', myColor(10,:), 'LineStyle', '--','LineWidth',lw);
        hold off
        axis('square'); xlim([1,10]);ylim([0,0.8]);yticks([0 0.4 0.8]); xticks([1 5 10]);xticklabels({'100','500','1000'});title('Heterogeneous SBMs');
        legend('G1','G2', 'G3','G1+G2+G3','20 Graphs','Location','SouthWest');
        set(gca,'FontSize',fs); 

        nexttile(tl)
        load('GEEFusionSim5Spec0.mat');
        plot(1:i,Acc1(:,ind),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',lw);hold on
        plot(1:i,Acc2(:,ind),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc3(:,ind),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',lw);
        plot(1:i,Acc123(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',lw);
        plot(1:i,Acc4(:,ind),'Color', myColor(10,:), 'LineStyle', '--','LineWidth',lw);
        hold off
        axis('square'); xlim([1,10]);ylim([0,0.8]);yticks([0 0.4 0.8]); xticks([1 5 10]);xticklabels({'100','500','1000'});title('Heterogeneous DC-SBMs');
        legend('G1','G2', 'G3','G1+G2+G3','20 Graphs','Location','SouthWest');
        set(gca,'FontSize',fs); 

        ylabel(tl,str1,'FontSize',fs)
        xlabel(tl,'Number of Vertices','FontSize',fs)
        %         set(gca,'FontSize',fs);
        F.wh=[8 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
end


if choice==103;%time figure
%         Spec=2;
        i=10;ind=1;lw=4;F.fname='FigFusion3';str1='Running Time';loc='East';
        myColor = brewermap(11,'RdYlGn'); myColor2 = brewermap(4,'RdYlBu');myColor(10,:)=myColor2(4,:);
%         myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
        fs=28;
        tl = tiledlayout(1,2);
        nexttile(tl)
        load('GEEFusionSim6.mat')
        errorbar(1:i,mean(time1,2),3*std(time1,[],2),'Color', myColor2(1,:),'LineStyle', '-','LineWidth',lw);hold on
        errorbar(1:i,mean(time2,2),3*std(time2,[],2),'Color', myColor2(2,:),'LineStyle', '-','LineWidth',lw);
        legend('One Graph','Three Graphs','Location','NorthWest');
        xlim([1,10]);xticks([1 5 10]);xticklabels({'3000','15000','30000'});title('Encoder Embedding');
        set(gca,'FontSize',fs); 
        axis('square'); 

        nexttile(tl)
        load('GEEFusionSim6.mat')
        errorbar(1:i,mean(time3,2),3*std(time3,[],2),'Color', myColor2(4,:),'LineStyle', '-','LineWidth',lw);hold on
        errorbar(1:i,mean(time4,2),3*std(time4,[],2),'Color', myColor2(3,:),'LineStyle', '-','LineWidth',lw);
        legend('One Graph','Three Graphs','Location','NorthWest');
        xlim([1,10]);xticks([1 5 10]);xticklabels({'3000','15000','30000'});title('Spectral Embedding');
        set(gca,'FontSize',fs); 
        ylabel(tl,'Running Time (s)','FontSize',fs)
        xlabel(tl,'Number of Vertices','FontSize',fs)
        axis('square'); 
        %         set(gca,'FontSize',fs);
        F.wh=[8 4]*2;
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

function Acc=result_acc(G)

[lim,rep]=size(G);
Acc=zeros(lim,10);
    for i=1:lim
        for r=1:rep
            Acc(i,1)=Acc(i,1)+G{i,r}{1,1}/rep;Acc(i,2)=Acc(i,2)+G{i,r}{1,2}/rep;Acc(i,3)=Acc(i,3)+G{i,r}{1,3}/rep;Acc(i,4)=Acc(i,4)+G{i,r}{1,4}/rep;Acc(i,5)=Acc(i,5)+G{i,r}{1,6}/rep;
            Acc(i,6)=Acc(i,6)+G{i,r}{4,1}/rep;Acc(i,7)=Acc(i,7)+G{i,r}{4,2}/rep;Acc(i,8)=Acc(i,8)+G{i,r}{4,3}/rep;Acc(i,9)=Acc(i,9)+G{i,r}{4,4}/rep;Acc(i,10)=Acc(i,10)+G{i,r}{4,6}/rep;
        end
    end
