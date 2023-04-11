function simDimension(choice,rep)

if nargin<2
    rep=2;
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
% 



if choice==1 || choice==2 || choice ==3 || choice==4 || choice==5 || choice ==6  % 3-dim signal, all signal, no-signal
    lim=3;G1=cell(lim,rep);G2=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'Normalize',false);
    optsE = opts; optsE.Elbow=1;
    for i=1:lim
        for r=1:rep
            n=200*i
            [Dis,Label]=simGenerate(300+(choice-1)+(choice>3)*7,n,5,1);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
        end
    end
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,1}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,2}/rep;Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,3}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,1}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,2}/rep;Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,3}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,1}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,2}/rep;Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,3}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,1}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,2}/rep;Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,3}/rep;
        end
    end
    save(strcat('GEEElbowSim',num2str(choice),'.mat'),'choice','Acc1','Acc2');
    [mean(Acc1);mean(Acc2)]
%     [std(Acc1);std(Acc2)]
end

if choice>6 && choice <100
    switch choice
        case 7
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols;
        case 8
            load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0);
        case 9
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
        case 10
            load('Wiki_Data.mat');G1=TE; G2=TF;
        case 11
           load('Cora.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30,'Normalize',false);
    optsE = opts; optsE.Elbow=1;
    Acc1=zeros(rep,6);Acc2=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,2};Acc1(i,4)=tmp{1,1};Time1(i,1)=tmp{4,2};Time1(i,4)=tmp{4,1};
        tmp=GraphEncoderEvaluate(G2,Label,opts);Acc1(i,2)=tmp{1,2};Acc1(i,5)=tmp{1,1};Time1(i,2)=tmp{4,2};Time1(i,5)=tmp{4,1};
        tmp=GraphEncoderEvaluate({G1,G2},Label,opts);Acc1(i,3)=tmp{1,2};Acc1(i,6)=tmp{1,1};Time1(i,3)=tmp{4,2};Time1(i,6)=tmp{4,1};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,2};Acc2(i,4)=tmp{1,1};Time2(i,1)=tmp{4,2};Time2(i,4)=tmp{4,1};
        tmp=GraphEncoderEvaluate(G2,Label,optsE);Acc2(i,2)=tmp{1,2};Acc2(i,5)=tmp{1,1};Time2(i,2)=tmp{4,2};Time2(i,5)=tmp{4,1};
        tmp=GraphEncoderEvaluate({G1,G2},Label,optsE);Acc2(i,3)=tmp{1,2};Acc2(i,6)=tmp{1,1};Time2(i,3)=tmp{4,2};Time2(i,6)=tmp{4,1};
    end
    save(strcat('GEEElbow',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end