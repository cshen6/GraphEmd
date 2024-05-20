function simDistance2(choice,rep,spec)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<3
    spec=0;
end
if nargin<2
    rep=3;
end
% Figure 1 SBM
if choice==1 || choice==2
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(18+(choice-1)*10,n,1);
            indices = crossvalind('Kfold',Label,3);
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
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc12','Acc13','Acc23','Acc123');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc12);mean(Acc13);mean(Acc23);mean(Acc123)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc12);std(Acc13);std(Acc23);std(Acc123)]
end

if choice==3
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G4=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(11,n,5);
            Dis1=simGenerate(11,n,5);
            Dis2=simGenerate(11,n,5);
            Dis3=simGenerate(11,n,5);
            Dis4=simGenerate(11,n,5);
            indices = crossvalind('Kfold',Label,3);
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

if choice==6
    load('Wiki_Data.mat'); Label=Label+1;
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',1,'GNN',0,'knn',5,'dim',30);
    % opts2=opts;
    % opts2.deg=1;opts2.ASE=0;
    Acc1=zeros(rep,13);Acc2=zeros(rep,13);Acc3=zeros(rep,13);Time1=zeros(rep,13);Time2=zeros(rep,13);Time3=zeros(rep,13);
    ind=1;ind2=2;ind3=3;%1: weighted LDA, 2: NN, 3: LDA. 4.ASE*NN, 5. ASE*LDA 
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,3);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(1-TE,Label,opts);Acc1(i,1)=tmp{1,ind};Acc2(i,1)=tmp{1,ind2};Acc3(i,1)=tmp{1,ind3};Time1(i,1)=tmp{4,2};Time2(i,1)=tmp{4,ind2};Time3(i,1)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(1-TF,Label,opts);Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(GE,Label,opts);Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
        tmp=GraphEncoderEvaluate(GF,Label,opts);Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Acc3(i,4)=tmp{1,ind3};Time1(i,4)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};Time3(i,4)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({1-TE,1-TF},Label,opts);Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Acc3(i,5)=tmp{1,ind3};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};Time3(i,5)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({GE,GF},Label,opts);Acc1(i,6)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Acc3(i,6)=tmp{1,ind3};Time1(i,6)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};Time3(i,6)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({GE,1-TE},Label,opts);Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,ind2};Acc3(i,7)=tmp{1,ind3};Time1(i,7)=tmp{4,ind};Time2(i,7)=tmp{4,ind2};Time3(i,7)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({GF,1-TF},Label,opts);Acc1(i,8)=tmp{1,ind};Acc2(i,8)=tmp{1,ind2};Acc3(i,8)=tmp{1,ind3};Time1(i,8)=tmp{4,ind};Time2(i,8)=tmp{4,ind2};Time3(i,8)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({1-TE,1-TF,GE},Label,opts);Acc1(i,9)=tmp{1,ind};Acc2(i,9)=tmp{1,ind2};Acc3(i,9)=tmp{1,ind3};Time1(i,9)=tmp{4,ind};Time2(i,9)=tmp{4,ind2};Time3(i,9)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({1-TE,1-TF,GF},Label,opts);Acc1(i,10)=tmp{1,ind};Acc2(i,10)=tmp{1,ind2};Acc3(i,10)=tmp{1,ind3};Time1(i,10)=tmp{4,ind};Time2(i,10)=tmp{4,ind2};Time3(i,10)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({GE,GF,1-TE},Label,opts);Acc1(i,11)=tmp{1,ind};Acc2(i,11)=tmp{1,ind2};Acc3(i,11)=tmp{1,ind3};Time1(i,11)=tmp{4,ind};Time2(i,11)=tmp{4,ind2};Time3(i,11)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({GE,GF,1-TF},Label,opts);Acc1(i,12)=tmp{1,ind};Acc2(i,12)=tmp{1,ind2};Acc3(i,12)=tmp{1,ind3};Time1(i,12)=tmp{4,ind};Time2(i,12)=tmp{4,ind2};Time3(i,12)=tmp{4,ind3};
%         tmp=GraphEncoderEvaluate({1-TE,1-TF,GE,GF},Label,opts);Acc1(i,13)=tmp{1,ind};Acc2(i,13)=tmp{1,ind2};Acc3(i,13)=tmp{1,ind3};Time1(i,13)=tmp{4,ind};Time2(i,13)=tmp{4,ind2};Time3(i,13)=tmp{4,ind3};
    end
    save(strcat('GEEDistanceWikiSpec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
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

if choice>=10 && choice<=13
    switch choice
        case 10
           load('Cora.mat');
        case 11
            load('citeseer.mat');
        case 12
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols;
        case 13
            load('COIL-RAG.mat'); 
%         case 14
%             load('COX2.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));
%         case 15
%             load('DHFR.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));
%         case 16
%             load('BZR.mat');K2=5;Dist2='sqeuclidean';Dist1='euclidean';D = squareform(pdist(X, Dist1));\
%         case 12
%             load('AIDS.mat');K2=30;Dist2='sqeuclidean';Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));
    end
    dimSplit=zeros(size(X,2),1);
    for i=1:size(X,2)
       [~,dimSplit(i)]=DCorFastTest(X(:,i),Label);
    end
    dimSplit=(dimSplit<0.05);
%     dimSplit=false(size(X,2),1);dimSplit(1:300)=true;
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'eval',0,'Matrix',1);
    Acc1=zeros(rep,5);Acc2=zeros(rep,5);Acc3=zeros(rep,5);Time1=zeros(rep,5);Time2=zeros(rep,5); Time3=zeros(rep,5);DistChoice='spearman';
    ind=3;ind2=2;ind3=1;
    D=cell(3,1);
    D{1} = 1-squareform(pdist(X, 'spearman'));
    a = 1-squareform(pdist(X(:,dimSplit), 'spearman'));a(isnan(a))=0;D{2}=a;
    a = 1-squareform(pdist(X(:,~dimSplit), 'spearman'));a(isnan(a))=0;D{3}=a;
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices; trn = ~(indices == i);
       tmp=AttributeEvaluate(X,Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
       tmp=GraphEncoderEvaluate({Edge,D{1}},Label,opts); Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate(Edge,Label,opts);Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate({Edge,D{2}},Label,opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Acc3(i,4)=tmp{1,ind3};Time1(i,4)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};Time3(i,4)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate({Edge,D{1},D{2},D{3}},Label,opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Acc3(i,5)=tmp{1,ind3};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};Time3(i,5)=tmp{4,ind3};
%        tmp=GraphEncoderEvaluate(D(1:s,1),Label,opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,6)=tmp{1,ind};Acc2(i,4)=tmp{1,6};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,4};
       %tmp=GraphEncoderEvaluate(D,Label,opts,X); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,4};
       %tmp=GraphEncoderEvaluate({Edge,D},Label,opts); Acc1(i,6)=tmp{1,ind};Acc2(i,6)=tmp{1,4};Time1(i,6)=tmp{4,ind};Time2(i,6)=tmp{4,4};
       %tmp=GraphEncoderEvaluate({Edge,D},Label,opts,X); Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,4};Time1(i,7)=tmp{4,ind};Time2(i,7)=tmp{4,4};
%        if spec==0
% %            tmp=AttributeEvaluate(tmpZ,Label,indices); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Time1(i,7)=tmp{4,2};
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X(:,dim)); Acc1(i,8)=tmp{1,ind};Time1(i,8)=tmp{4,2};
%        end
%        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,4};
%        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};
%        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
%        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEEDistance',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
     [std(Acc1);std(Acc2);std(Acc3);std(Time1);std(Time2);std(Time3)]
end

if choice>=20 && choice<=29
    switch choice
        case 20
           load("faceYaleB_32x32");dimhei=32;dimwid=dimhei;%LDA works 0.2
%         case 21
% %             [fea,gnd]=simGenerateDis(1,100,5,2);
%         case 25
%             load("Yale_64x64"); dimhei=64;dimwid=dimhei;%LDA works 0.2
%         case 25
%            load("faceORL_64x64");dimhei=64;dimwid=dimhei;
        case 21
            load("COIL20");dimhei=32;dimwid=dimhei;%LDA works 3.0
        case 22
            load("facePIE_32x32"); dimhei=32;dimwid=dimhei;%LDA works 2.0%
        case 23
            load("USPS"); dimhei=16;dimwid=dimhei;%kNN works? 6%
%         case 24
%             load("MNIST2"); dimhei=28;dimwid=dimhei;%kNN works? 6%
        case 24
            load("umist");
        case 25
            load("binaryalpha");
        case 27
            load("binaryalpha");
%         case 25
%             load("TDT2"); %kNN works? 6%
%         case 26
%             load("Reuters21578"); %kNN works? 6%
%         case 27
%             load("Isolet"); %kNN works? 6%
    end
    X=fea(1:min(20000,length(gnd)),:);Label=gnd(1:min(20000,length(gnd)));
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'eval',0,'Matrix',1,'Normalization',true);
    Acc1=zeros(rep,5);Acc2=zeros(rep,5);Acc3=zeros(rep,5);Time1=zeros(rep,5);Time2=zeros(rep,5); Time3=zeros(rep,5);DistChoice='spearman';
    ind=3;ind2=2;ind3=1;
    dim=size(X,2);
    dimSplit=cell(1,3);
    dimSplit{1}=1:dim;
    dimSplit{2}=false(dim,2);dimSplit{2}(1:floor(dim/2),1)=1;dimSplit{2}(floor(dim/2)+1:end,2)=1;
    dimSplit{3}=false(dim,4); tmp=false(dimhei,dimhei);
    tmp1=tmp;tmp1(1:floor(dimhei/2),1:floor(dimwid/2))=1;dimSplit{3}(:,1)=reshape(tmp1,dim,1);
    tmp1=tmp;tmp1(floor(dimhei/2)+1:end,1:floor(dimwid/2))=1;dimSplit{3}(:,2)=reshape(tmp1,dim,1);
    tmp1=tmp;tmp1(1:floor(dimhei/2),floor(dimwid/2)+1:end)=1;dimSplit{3}(:,3)=reshape(tmp1,dim,1);
    tmp1=tmp;tmp1(floor(dimhei/2)+1:end,floor(dimwid/2)+1:end)=1;dimSplit{3}(:,4)=reshape(tmp1,dim,1);    
    split=3;s=1;D=cell(1); %2 or 3 work equally well
    for j=1:split
        if j==1
            D{s,1} = 1-squareform(pdist(X(:,dimSplit{j}), DistChoice));
%             D{s,1} = X(:,dimSplit{j})*X(:,dimSplit{j})';
            s=s+1;
        else
            for i=1:size(dimSplit{j},2)
               D{s,1} = 1-squareform(pdist(X(:,dimSplit{j}(:,i)), DistChoice));
%                 D{s,1}=X(:,dimSplit{j}(:,i))*X(:,dimSplit{j}(:,i))';
                s=s+1;
            end
        end
    end
% D=cell(3,1);
%     D{1} = 1-squareform(pdist(X, 'cosine'));
%     D{2} = squareform(pdist(X, 'euclidean'));D{2}=max(max(D{2}))-D{2};
%     D{3} = 1-squareform(pdist(X, 'spearman'));
%     D4 = DCorInput(X,'hsic');
%     if spec>0
%         Z1=cmds(D1);
%         Z2=cmds(D1);
%     end
    
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,4);
       opts.indices=indices;
       tmp=AttributeEvaluate(X,Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
       tmp=GraphEncoderEvaluate(D{1},Label,opts); Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate({D{2},D{3}},Label,opts); Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,ind2};Acc3(i,3)=tmp{1,ind3};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,ind2};Time3(i,3)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate({D{1},D{2},D{3}},Label,opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Acc3(i,4)=tmp{1,ind3};Time1(i,4)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};Time3(i,4)=tmp{4,ind3};
       tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Acc3(i,5)=tmp{1,ind3};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};Time3(i,5)=tmp{4,ind3};
%        tmp=GraphEncoderEvaluate(1-D1,Label,opts); Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(M2-D2,Label,opts); Acc1(i,3)=tmp{1,ind};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,ind};Time2(i,3)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(D4,Label,opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,4};Time1(i,4)=tmp{4,ind};Time2(i,4)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(1-D1,Label,opts,X); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(M2-D2,Label,opts,X); Acc1(i,6)=tmp{1,ind};Acc2(i,6)=tmp{1,4};Time1(i,6)=tmp{4,ind};Time2(i,6)=tmp{4,4};
%        tmp=GraphEncoderEvaluate(D4,Label,opts,X); Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,4};Time1(i,7)=tmp{4,ind};Time2(i,7)=tmp{4,4};
%        tmp=GraphEncoderEvaluate({1-D1,M2-D2},Label,opts); Acc1(i,8)=tmp{1,ind};Acc2(i,8)=tmp{1,4};Time1(i,8)=tmp{4,ind};Time2(i,8)=tmp{4,4};
%        tmp=GraphEncoderEvaluate({1-D1,M2-D2},Label,opts,X); Acc1(i,9)=tmp{1,ind};Acc2(i,9)=tmp{1,4};Time1(i,9)=tmp{4,ind};Time2(i,9)=tmp{4,4};
%        tmp=GraphEncoderEvaluate({1-D1,M2-D2,D4},Label,opts); Acc1(i,10)=tmp{1,ind};Acc2(i,10)=tmp{1,4};Time1(i,10)=tmp{4,ind};Time2(i,10)=tmp{4,4};
%        tmp=GraphEncoderEvaluate({1-D1,M2-D2,D4},Label,opts,X); Acc1(i,11)=tmp{1,ind};Acc2(i,11)=tmp{1,4};Time1(i,11)=tmp{4,ind};Time2(i,11)=tmp{4,4};
% %        tmp=GraphEncoderEvaluate({D1,D2},Label,opts,X); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};Time1(i,5)=tmp{4,ind};Time2(i,5)=tmp{4,4};
%        if spec>0
%            tmp=AttributeEvaluate(Z1,Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
%            tmp=AttributeEvaluate(Z2,Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
%            tmp=AttributeEvaluate([Z1,Z2],Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
%            tmp=AttributeEvaluate([Z1,Z2,X],Label,indices); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
%        end
%        if spec==0
% %            tmp=AttributeEvaluate(tmpZ,Label,indices); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Time1(i,7)=tmp{4,2};
%            tmp=GraphEncoderEvaluate(Edge,Label,opts,X(:,dim)); Acc1(i,8)=tmp{1,ind};Time1(i,8)=tmp{4,2};
%        end
%        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,ind};Acc2(i,4)=tmp{1,4};
%        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,ind};Acc2(i,5)=tmp{1,4};
%        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
%        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,ind};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEEDistance',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
     [std(Acc1);std(Acc2);std(Acc3);std(Time1);std(Time2);std(Time3)]
end
% 
% if choice>=100 
%     switch choice
%         case 100
%            load("faceYaleB_32x32");dimhei=32;dimwid=dimhei;%LDA works 0.2
% %         case 21
% % %             [fea,gnd]=simGenerateDis(1,100,5,2);
% %         case 25
% %             load("Yale_64x64"); dimhei=64;dimwid=dimhei;%LDA works 0.2
% %         case 25
% %            load("faceORL_64x64");dimhei=64;dimwid=dimhei;
%         case 101
%             load("COIL20");dimhei=32;dimwid=dimhei;%LDA works 3.0
% %         case 22
% %             load("facePIE_32x32"); dimhei=32;dimwid=dimhei;%LDA works 2.0%
% %         case 23
% %             load("USPS"); dimhei=16;dimwid=dimhei;%kNN works? 6%
% %         case 24
% %             load("MNIST2"); dimhei=28;dimwid=dimhei;%kNN works? 6%
% %         case 26
% %             load("umist");
% %         case 26
% %             load("binaryalpha");
% %         case 25
% %             load("TDT2"); %kNN works? 6%
% %         case 26
% %             load("Reuters21578"); %kNN works? 6%
% %         case 27
% %             load("Isolet"); %kNN works? 6%
%     end
% %     image(reshape(fea(1,:),32,32))
%     X=fea(1:min(20000,length(gnd)),:);Label=gnd(1:min(20000,length(gnd)));
%     opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'eval',0,'Matrix',1);
%     Acc1=zeros(rep,5);Acc2=zeros(rep,5);Acc3=zeros(rep,5);Time1=zeros(rep,5);Time2=zeros(rep,5); Time3=zeros(rep,5);DistChoice='spearman';
%     ind=3;ind2=2;ind3=1;
%     dim=size(X,2);
%     dimSplit=cell(1,3);
%     dimSplit{1}=1:dim;
%     dimSplit{2}=false(dim,2);dimSplit{2}(1:floor(dim/2),1)=1;dimSplit{2}(floor(dim/2)+1:end,2)=1;
%     dimSplit{3}=false(dim,4); tmp=false(dimhei,dimhei);
%     tmp1=tmp;tmp1(1:floor(dimhei/2),1:floor(dimwid/2))=1;dimSplit{3}(:,1)=reshape(tmp1,dim,1);
%     tmp1=tmp;tmp1(floor(dimhei/2)+1:end,1:floor(dimwid/2))=1;dimSplit{3}(:,2)=reshape(tmp1,dim,1);
%     tmp1=tmp;tmp1(1:floor(dimhei/2),floor(dimwid/2)+1:end)=1;dimSplit{3}(:,3)=reshape(tmp1,dim,1);
%     tmp1=tmp;tmp1(floor(dimhei/2)+1:end,floor(dimwid/2)+1:end)=1;dimSplit{3}(:,4)=reshape(tmp1,dim,1);    
%     split=3;s=1;D=cell(1); %2 or 3 work equally well
%     for j=1:split
%         if j==1
%             D{s,1} = 1-squareform(pdist(X(:,dimSplit{j}), DistChoice));
%             s=s+1;
%         else
%             for i=1:size(dimSplit{j},2)
%                 D{s,1} = 1-squareform(pdist(X(:,dimSplit{j}(:,i)), DistChoice));
%                 s=s+1;
%             end
%         end
%     end
%     D1=D{1};
%     D2=D(1:3);
%     D3=D{:};
%     [Z1,out1]=GraphEncoder(D1,gnd);
%     c1=kmeans(Z1,max(gnd)); RandIndex(c1,gnd)
%     [Z2,out2]=GraphEncoder(D2,gnd); Z2=horzcat(Z2{:});
%     c2=kmeans(Z2,max(gnd)); RandIndex(c2,gnd)
%     [Z3,out3]=GraphEncoder(D3,gnd); Z3=horzcat(Z3{:});
%     c3=kmeans(Z3,max(gnd)); RandIndex(c3,gnd)
%     zz1=pca(Z1','numComponents',2);
%     hold on
%     plot(zz1(gnd==1,1),zz1(gnd==1,2),'r.');
%     plot(zz1(gnd==2,1),zz1(gnd==2,2),'b.');
% end

if choice>=30 && choice<=40
    switch choice
        case 30
           load("mnist");
%         case 21
%            load("faceORL_32x32");
        %case 22
         %   load("faceORL_64x64");
        case 31
            load("cifar10");
    end
    tic
    discrimType='pseudolinear';err=zeros(5,1);time=zeros(5,1);
    %mdl=fitcdiscr(X(idx,:),Y(idx),'discrimType',discrimType);
    mdl=fitcknn(X(idx,:),Y(idx),'Distance','Euclidean','NumNeighbors',5);
%     mdl=fitcensemble(X(idx,:),Y(idx),'Method','Bag');
    tt=predict(mdl,X(~idx,:));
    err(1)=mean(Y(~idx)~=tt);
    time(1)=toc;
    
    tic
    split=3;s=1;D=cell(sum(1:split),1); dim=size(X,2);%2 or 3 work equally well
    for j=1:split
        for i=1:j
            dimInd=floor(dim/j);
            D1 = squareform(pdist(X(:,dimInd*(i-1)+1:dimInd*i), 'spearman'));
            D{s,1}=1-D1;
            s=s+1;
        end
    end
    tmpTime=toc;

    tic
    Y2=Y;Y2(~idx)=0;
    Z=GraphEncoder(D{1},Y2);
    mdl=fitcknn(Z(idx,:),Y(idx),'Distance','Euclidean','NumNeighbors',5);
    %mdl=fitcdiscr(Z(idx,:),Y(idx),'discrimType',discrimType);
%     mdl=fitcensemble(Z(idx,:),Y(idx),'Method','Bag');
    tt=predict(mdl,Z(~idx,:));
    err(2)=mean(Y(~idx)~=tt);
    time(2)=toc+tmpTime;

    Y2=Y;Y2(~idx)=0;
    Z=GraphEncoder({D{2},D{3}},Y2);
    mdl=fitcknn(Z(idx,:),Y(idx),'Distance','Euclidean','NumNeighbors',5);
    %mdl=fitcdiscr(Z(idx,:),Y(idx),'discrimType',discrimType);
%     mdl=fitcensemble(Z(idx,:),Y(idx),'Method','Bag');
    tt=predict(mdl,Z(~idx,:));
    err(3)=mean(Y(~idx)~=tt);
    time(3)=toc+tmpTime;

     Y2=Y;Y2(~idx)=0;
    Z=GraphEncoder(D,Y2);
    mdl=fitcknn(Z(idx,:),Y(idx),'Distance','Euclidean','NumNeighbors',5);
    %mdl=fitcdiscr(Z(idx,:),Y(idx),'discrimType',discrimType);
%     mdl=fitcensemble(Z(idx,:),Y(idx),'Method','Bag');
    tt=predict(mdl,Z(~idx,:));
    err(4)=mean(Y(~idx)~=tt);
    time(4)=toc+tmpTime;

    save(strcat('GEEDistance',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','err','time');
%     [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
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
