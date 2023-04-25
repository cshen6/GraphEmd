function simDimension(choice,rep)

if nargin<2
    rep=20;
end
% 
% n=3000;k=10;type=300;
% [Dis,Label]=simGenerate(type,n,k);
% Z=GraphEncoder(Dis,Label);
% q=getElbow(std(Z),1)
% 
% 
% [Dis,Label]=simGenerate(28,3000,4);
% Z=GraphEncoder(Dis,Label);
% Z=horzcat(Z{:});
% score=std(Z);
% [~,dim]=sort(score,'descend');
% [~,Z2]=pca(Z,'NumComponent',3);
% ind1=(Label==1);ind2=(Label==2);ind3=(Label==3);ind4=(Label==4);
% 
% tl = tiledlayout(1,2);myColor = brewermap(4,'RdYlGn'); 
% nexttile(tl)
% scatter3(Z2(ind1,1), Z2(ind1,2),Z2(ind1,3),20,myColor(1,:),'filled');hold on
% scatter3(Z2(ind2,1), Z2(ind2,2),Z2(ind2,3),20,myColor(2,:),'filled');
% scatter3(Z2(ind3,1), Z2(ind3,2),Z2(ind3,3),20,myColor(3,:),'filled');
% scatter3(Z2(ind4,1), Z2(ind4,2),Z2(ind4,3),20,myColor(4,:),'filled');
% nexttile(tl)
% Z2=Z(:,dim(1:3));
% scatter3(Z2(ind1,1), Z2(ind1,2),Z2(ind1,3),20,myColor(1,:),'filled');hold on
% scatter3(Z2(ind2,1), Z2(ind2,2),Z2(ind2,3),20,myColor(2,:),'filled');
% scatter3(Z2(ind3,1), Z2(ind3,2),Z2(ind3,3),20,myColor(3,:),'filled');
% scatter3(Z2(ind4,1), Z2(ind4,2),Z2(ind4,3),20,myColor(4,:),'filled');
% % Z=GraphEncoder(Dis,Label(randperm(1000)));
% % Z=horzcat(Z{:});
% % std(Z)
% 
% 
% 
% [Dis,Label]=simGenerate(18,3000,4);
% Dis={Dis{1},Dis{2}};
% Z=GraphEncoder(Dis,Label);
% Z=horzcat(Z{:});
% score=std(Z);
% [~,dim]=sort(score,'descend');
% [~,Z2]=pca(Z,'NumComponent',2);
% ind1=(Label==1);ind2=(Label==2);ind3=(Label==3);
% 
% tl = tiledlayout(1,2);myColor = brewermap(4,'RdYlGn'); 
% nexttile(tl)
% scatter(Z2(ind1,1), Z2(ind1,2),20,myColor(1,:),'filled');hold on
% scatter(Z2(ind2,1), Z2(ind2,2),20,myColor(2,:),'filled');
% scatter(Z2(ind3,1), Z2(ind3,2),20,myColor(3,:),'filled');
% nexttile(tl)
% Z2=Z(:,dim(1:2));
% scatter(Z2(ind1,1), Z2(ind1,2),20,myColor(1,:),'filled');hold on
% scatter(Z2(ind2,1), Z2(ind2,2),20,myColor(2,:),'filled');
% scatter(Z2(ind3,1), Z2(ind3,2),20,myColor(3,:),'filled');

if choice==1 || choice==2 || choice ==3 || choice==4 || choice==5 || choice ==6 || choice ==7 || choice ==8  % top 3; all; none; none; repeat for DC-SBM
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);dim=20; ind=2;ind2=3;
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=true;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,4);
    for i=1:lim
        for r=1:rep
            n=300*i
            [Dis,Label]=simGenerate(300+(choice-1)+(choice>4)*6,n,dim,1);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
            [Z,out]=GraphEncoder(Dis,Label,0,optsE);
            Acc3(i,1)=Acc3(i,1)+sum(out.DimScore(1:3))/3/rep;
            Acc3(i,2)=Acc3(i,2)+sum(out.DimChoice(1:3))/3/rep;
            Acc3(i,3)=Acc3(i,3)+sum(out.DimScore(4:end))/(dim-3)/rep;
            Acc3(i,4)=Acc3(i,4)+sum(out.DimChoice(4:end))/(dim-3)/rep;
        end
    end
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,ind2}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,ind}/rep;%Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,ind2}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,ind2}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,ind}/rep;%Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,ind2}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,ind2}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,ind2}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Z','out')
    [mean(Acc1);mean(Acc2)]
    out.DimScore
%     [std(Acc1);std(Acc2)]
end

if choice==9 || choice==10
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep); ind=2;ind2=3;n=5000;
    if choice==9
        type=300;
    else
        type=310;
    end
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=true;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,4);
    for i=1:lim
        for r=1:rep
            dim=5*i
            [Dis,Label]=simGenerate(type,n,dim,1);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
            [Z,out]=GraphEncoder(Dis,Label,0,optsE);
            Acc3(i,1)=Acc3(i,1)+sum(out.DimScore(1:3))/3/rep;
            Acc3(i,2)=Acc3(i,2)+sum(out.DimChoice(1:3))/3/rep;
            Acc3(i,3)=Acc3(i,3)+sum(out.DimScore(4:end))/(dim-3)/rep;
            Acc3(i,4)=Acc3(i,4)+sum(out.DimChoice(4:end))/(dim-3)/rep;
        end
    end
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,ind2}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,ind}/rep;%Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,ind2}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,ind2}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,ind}/rep;%Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,ind2}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,ind2}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,ind2}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Z','out')
    [mean(Acc1);mean(Acc2)]
%     out.DimScore
end

if choice>=11 && choice <=18
    switch choice
        case 11
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols; %reduce to 2/3 for G2
        case 12
            load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0); %reduced to 70% for both graph
        case 13
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
        case 14
            load('Wiki_Data.mat');G1=1-TE; G2=GE; Label=Label+1;
        case 15
           load('Cora.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
        case 16
           load('citeseer.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced 1 dim in G1
        case 17
           load('protein.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
        case 18
           load('COIL-RAG.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced to 88/100 dim in G2.
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=1; ind=2;ind2=3;
    Acc1=zeros(rep,6);Acc2=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,4)=tmp{1,ind2};Time1(i,1)=tmp{4,ind};Time1(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G2,Label,opts);Acc1(i,2)=tmp{1,ind};Acc1(i,5)=tmp{1,ind2};Time1(i,2)=tmp{4,ind};Time1(i,5)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate({G1,G2},Label,opts);Acc1(i,3)=tmp{1,ind};Acc1(i,6)=tmp{1,ind2};Time1(i,3)=tmp{4,ind};Time1(i,6)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G2,Label,optsE);Acc2(i,2)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Time2(i,2)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate({G1,G2},Label,optsE);Acc2(i,3)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder({G1,G2},Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out')
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
%     [~,out]=GraphEncoder(G1,Label,0,optsE);
    sum(out(1).DimScore>1)/length(out(1).DimScore)
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice>=19 && choice <=26
    switch choice
        case 19
            load('adjnoun.mat');G1=Edge+1; Label=Y; %kept 1/2 dimension
        case 20
            load('email.mat');G1=Edge;Label=Y; %kept 36/42 dimension
        case 21
            load('Gene.mat');G1=Edge;Label=Y;
        case 22
            load('IIP.mat');G1=Edge;Label=Y; %kept 2/3 dimension
        case 23
           load('LastFM.mat');G1=Edge;Label=Y; %kept 17/18 dimension
        case 24
           load('polblogs.mat');G1=Edge;Label=Y;
        case 25
           load('pubmed.mat');G1=Edge;
        case 26
           load('smartphone.mat');G1=Edge; %kept 49/71 dimension
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=1;ind=2;ind2=3;
    Acc1=zeros(rep,6);Acc2=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,4)=tmp{1,ind2};Time1(i,1)=tmp{4,ind};Time1(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder(G1,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    sum(out(1).DimScore>1)/length(out(1).DimScore)
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==27 %kept 20 out of 39 dimensions
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=1;ind=2;ind2=3;
    load('anonymized_msft.mat')
    Label=label;rep=1;i=1;Acc1=zeros(rep,6);Acc2=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);
    indices = crossvalind('Kfold',Label,5);
    opts.indices=indices;optsE.indices=indices;
%     tmp=GraphEncoderEvaluate(G{6},Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,4)=tmp{1,ind2};Time1(i,1)=tmp{4,ind};Time1(i,4)=tmp{4,ind2};
%     tmp=GraphEncoderEvaluate(G{6},Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
%     tmp=GraphEncoderEvaluate(G{12},Label,opts);Acc1(i,2)=tmp{1,ind};Acc1(i,5)=tmp{1,ind2};Time1(i,2)=tmp{4,ind};Time1(i,5)=tmp{4,ind2};
%     tmp=GraphEncoderEvaluate(G{12},Label,optsE);Acc2(i,2)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Time2(i,2)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};
%     tmp=GraphEncoderEvaluate(G{18},Label,opts);Acc1(i,3)=tmp{1,ind};Acc1(i,6)=tmp{1,ind2};Time1(i,3)=tmp{4,ind};Time1(i,6)=tmp{4,ind2};
%     tmp=GraphEncoderEvaluate(G{18},Label,optsE);Acc2(i,3)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};
    tic
    [Z,out]=GraphEncoder(G,label,0,opts);
    time=toc;
    %save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','Z','out');
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','out','time');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    DimScore=zeros(1,39);thres=1;
    DimChoice=(DimScore==1);
    for i=1:24
        DimScore=DimScore+out(i).DimScore/24;
        DimChoice=DimChoice | (out(i).DimScore>thres);
    end
%     sum(out(1).DimScore>1)/length(out(1).DimScore)
end

if choice==28
    load('anonymized_msft.mat');
    load('GEEDimension27.mat');tt=1;
%     [~,Z3]=pca(Z(:,DimChoice,tt),'numComponents',3,'Centered',false);
%     [~,Z4]=pca(Z(:,:,tt),'numComponents',3,'Centered',false);
    [Z3,umap,clusterIdentifiers,extras]=run_umap(Z(:,DimChoice,tt),'n_components',3);
    [Z4,umap,clusterIdentifiers,extras]=run_umap(Z(:,:,tt),'n_components',3);
    maxK=39;
    myColor = brewermap(maxK,'RdYlGn');
    tl = tiledlayout(1,2);
    nexttile(tl)
    i=1;
    ind=(label==i);scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
    hold on
    for i=2:maxK
        ind=(label==i);
        scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
    end
    title('Full GEE * UMAP')
    hold off
    nexttile(tl)
    i=1;
    ind=(label==i);scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
    hold on
    for i=2:maxK
        ind=(label==i);
        scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
        hold on
    end
    hold off
    title('Principal GEE * UMAP')
end