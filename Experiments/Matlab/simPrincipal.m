function simPrincipal(choice,rep)

if nargin<2
    rep=2;
end
norma=true;thres1=0.7;
if choice==1 || choice==2 || choice ==3 % top 3; all; none; none; repeat for DC-SBM
    lim=20;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);dim=20; ind=3;ind2=2;
    opts = struct('Adjacency',1,'Laplacian',0,'Normalize',norma,'Discriminant',0,'Principal',1,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Principal=2;
%     optsE2=optsE; optsE2.Dimension=2;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);
    type=300;
    switch choice
        case 1
            type=300; nn=250;
        case 2 
            type=310; nn=250;
%         case 3
%             type=301;
        case 3
            type=320; nn=250;
    end
    for i=1:lim
        for r=1:rep
            n=nn*i
            [Dis,Label]=simGenerate(type,n,dim,0);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
%             G3{i,r}=GraphEncoderEvaluate(Dis,Label,optsE2);
            [Z,out]=GraphEncoder(Dis,Label,optsE);out1=out.comChoice;out2=out.comScore;
            Acc3(i,1)=Acc3(i,1)+sum(out1(1:3))/3/rep;
            Acc3(i,2)=Acc3(i,2)+sum(out2(1:3))/3/rep;
            Acc3(i,3)=Acc3(i,3)+sum(out1(4:end))/(dim-3)/rep;
            Acc3(i,4)=Acc3(i,4)+sum(out2(4:end))/(dim-3)/rep;
%             [Z,out]=GraphEncoder(Dis,Label,0,optsE2);
%             Acc3(i,5)=Acc3(i,5)+sum(out.Std(1:3))/3/rep;
%             Acc3(i,6)=Acc3(i,6)+sum(out.Std(4:end))/(dim-3)/rep;
        end
    end
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,ind2}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,ind}/rep;%Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,ind2}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,ind2}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,ind}/rep;%Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,ind2}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,ind2}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,ind2}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
%             Acc4(i,1)=Acc4(i,1)+G3{i,r}{1,ind2}/rep;Acc4(i,2)=Acc4(i,2)+G3{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
%             Acc4(i,4)=Acc4(i,4)+G3{i,r}{4,ind2}/rep;Acc4(i,5)=Acc4(i,5)+G3{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Z','out')
    [mean(Acc1);mean(Acc2);mean(Acc3)]
    sum(out.comChoice)
%     [std(Acc1);std(Acc2)]
end

if choice==4 || choice==5 || choice==6 
    lim=20;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);ind=3;ind2=2;n=5000;
    switch choice
        case 4
            type=300;
        case 5 
            type=310;
%         case 9
%             type=301;
        case 6
            type=320;
    end
    opts = struct('Adjacency',1,'Laplacian',0,'Normalize',norma,'Discriminant',0,'Principal',1,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Principal=3;
%     optsE2=optsE; optsE2.Dimension=2;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            dim=10*i
            [Dis,Label]=simGenerate(type,n,dim,0);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
%             G3{i,r}=GraphEncoderEvaluate(Dis,Label,optsE2);
            [Z,out]=GraphEncoder(Dis,Label,optsE);out1=out.comChoice;out2=out.comScore;
            Acc3(i,1)=Acc3(i,1)+sum(out1(1:3))/3/rep;
            Acc3(i,2)=Acc3(i,2)+sum(out2(1:3))/3/rep;
            Acc3(i,3)=Acc3(i,3)+sum(out1(4:end))/(dim-3)/rep;
            Acc3(i,4)=Acc3(i,4)+sum(out2(4:end))/(dim-3)/rep;
%             Acc3(i,5)=Acc3(i,5)+sum(out.Std(1:3))/3/rep;
%             Acc3(i,6)=Acc3(i,6)+sum(out.Std(4:end))/(dim-3)/rep;
        end
    end
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,ind2}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,ind}/rep;%Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,ind2}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,ind2}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,ind}/rep;%Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,ind2}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,ind2}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,ind2}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
%             Acc4(i,1)=Acc4(i,1)+G3{i,r}{1,ind2}/rep;Acc4(i,2)=Acc4(i,2)+G3{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
%             Acc4(i,4)=Acc4(i,4)+G3{i,r}{4,ind2}/rep;Acc4(i,5)=Acc4(i,5)+G3{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Z','out')
    [mean(Acc1);mean(Acc2);mean(Acc3)]
%     out.DimScore
end

% for i=30:41
% simDimension(i,100)
% end
if choice>=10 && choice <20
    %30-35
    rng("default")
    opts = struct('Adjacency',1,'Normalize',norma,'Laplacian',0,'Spectral',0,'Discriminant',0,'LDA',1,'GNN',0,'knn',0,'dim',30);
    ind=3;ind2=2; spectral=0;n2v=1;
    optsE = opts; optsE.Principal=3;optsE.Spectral=0;
    switch choice
        case 10
           load('citeseer.mat');X=edge2adj(Edge);G1=Edge; optsE.Principal=1;n2vstr='Citeseer';
        case 11
           load('Cora.mat');X=edge2adj(Edge);G1=Edge;n2vstr='Cora'; %ind=2;ind2=4;% 3 out of 5
        case 12
            load('email.mat');X=Adj;G1=Edge;Label=Y; n2vstr='email';%kept 39/42 dimension
        case 13
            load('IIP.mat');X=double(Adj+Adj'>0);G1=Edge;Label=Y; n2vstr='IIP';
            %         case 17
            %             load('COIL-RAG.mat');G1=Edge;
        case 14
            load('IMDB.mat');G1=Edge2;Label=Label2+1;n2vstr='IMDB2';
        case 15
           load('LastFM.mat');X=Adj;G1=Edge;Label=Y; n2vstr='lastfm';%optsE.Principal=2;%kept 17/18 dimension
        case 16
           load('Letter.mat');G1=Edge1;Label=Label1;n2vstr='letter1';%optsE.Principal=3; % 4/15
        case 17
            load('smartphone.mat');G1=Edge; X=edge2adj(G1); n2vstr='phone';%optsE.Principal=3;%kept 53/71 dimension
        case 18
            load('protein.mat');Dist1='cosine';G1=Edge;n2vstr='protein';
        case 19
            load('pubmed.mat');Dist1='cosine';G1=Edge;n2vstr='pubmed';
%         case 16
%             load('Wiki_Data.mat');G1=GFAdj; optsE.Principal=2;%kept all
%         case 34
%            load('anonymized_msft.mat');G1=G{1}; Label=label;
    end
    %     optsE2=optsE; optsE2.Dimension=2;
    if spectral==1
        tic
        %[Z]=UnsupGraph(X,max(Y)*5,length(Y));
        dim=30;
        [ZASE]=ASE(X,dim);
        tt1=toc;
    end
    if n2v==1
        ZNV=load('n2v.mat',n2vstr);
        ZNV=ZNV.(n2vstr);
    end

    Acc1=zeros(rep,4);Acc2=zeros(rep,4);Time1=zeros(rep,4);Time2=zeros(rep,4);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;optsE2.indices=indices;
        tmp=GraphEncoderEvaluate(X,Label,opts);Acc1(i,1)=tmp{1,ind};%Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind+2};Acc1(i,4)=tmp{1,ind2+2};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind+2};Time1(i,4)=tmp{4,ind2+2};
        tmp=GraphEncoderEvaluate(X,Label,optsE);Acc2(i,2)=tmp{1,ind};%Acc2(i,2)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};
        if spectral==1
            tmp=AttributeEvaluate(ZASE,Y,indices); %K=6
            Acc1(i,3)=tmp(1,1);
        end
        if n2v==1
            tmp=AttributeEvaluate(ZNV,Y,indices); %K=6
            Acc1(i,4)=tmp(1,1);
        end
%         tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc2(i,3)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder(G1,Label,optsE);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out');
    [mean(Acc1);mean(Acc2);std(Acc1);std(Acc2);mean(Time1);mean(Time2)] %GEE; ASE; PGEE; PCA
    [sum(out.comChoice),max(Label)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice>=20 && choice <30 %noise
    %30-35
    opts = struct('Adjacency',1,'Normalize',norma,'Laplacian',0,'Spectral',0,'Discriminant',false,'Principal',1,'LDA',1,'GNN',0,'knn',0,'dim',30);
    ind=3;ind2=1;
    optsE = opts; optsE.Principal=3;optsE.Spectral=0;
    switch choice
%        case 20
%            load('citeseer.mat');G1=Edge; %G1=Edge+1; Label=Y; %kept 1/2 dimension
%         case 25
%            load('Cora.mat');Dist1='cosine';G1=Edge; 
        case 20
            load('Gene.mat');G1=Edge;Label=Y;
%         case 21
%             load('IMDB.mat');G1=Edge1;Label=Label1+1;
%         case 22
%            load('Letter.mat');Dist1='cosine';G1=Edge1;Label=Label1;
        case 21
            load('polblogs.mat');G1=Edge;Label=Y;
        case 22
           load('protein.mat');Dist1='cosine';G1=Edge; 
        case 23
           load('pubmed.mat');Dist1='cosine';G1=Edge; 
%         case 24
%             load('Wiki_Data.mat');G1=GEAdj; %kept all
%         case 40
%            load('Letter');G1=Edge1;Label=Label1;
%         case 24
%             load('Letter.mat');G1=Edge1;Label=GraphID1;
%         case 25
%             load('COIL-RAG.mat');G1=Edge;Label=GraphID;
%         case 21
%             load('IMDB.mat');G1=Edge1;Label=GraphID1;
    end
%     optsE2=optsE; optsE2.Dimension=2;
    Acc1=zeros(rep,4);Acc2=zeros(rep,4);Time1=zeros(rep,4);Time2=zeros(rep,4);noise=randi(300,1,length(Label));K=max(Label);
    for i=1:length(Label)
        if noise(i)>270
            Label(i)=noise(i)-270+K;
        end
    end
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;optsE2.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind+2};Acc1(i,4)=tmp{1,ind2+2};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind+2};Time1(i,4)=tmp{4,ind2+2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};
%         tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc2(i,3)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder(G1,Label,optsE);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out');
    [mean(Acc1);mean(Acc2);std(Acc1);std(Acc2);mean(Time1);mean(Time2)] %GEE; ASE; PGEE; PCA
    [sum(out.comChoice),max(Label)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice>=30 && choice <40 %fusion
    opts = struct('Adjacency',1,'Normalize',norma,'Laplacian',0,'Spectral',0,'LDA',1,'Discriminant',true,'GNN',0,'knn',0,'dim',30);
    ind=3;ind2=1;%5 for Spectral
    optsE = opts; optsE.Principal=3;optsE.Spectral=0;
    switch choice
        case 30
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols; 
        case 31
            load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0); %reduced to 70% for both graph
        case 32
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
        case 33
            load('Wiki_Data.mat');G1=1-TE; G2=GE; Label=Label;
        case 34
           load('Cora.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); % 3 out of 5
        case 35
           load('citeseer.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced 1 dim in G1
        case 36
           load('protein.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
%         case 28
%            load('COIL-RAG.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced to 88/100 dim in G2.
    end
%     optsE2=optsE; optsE2.Dimension=2;
    Acc1=zeros(rep,6);Acc2=zeros(rep,6);Acc3=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);Time3=zeros(rep,6);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G2,Label,opts);Acc1(i,3)=tmp{1,ind};Acc1(i,4)=tmp{1,ind2};Time1(i,3)=tmp{4,ind};Time1(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate({G1,G2},Label,opts);Acc1(i,5)=tmp{1,ind};Acc1(i,6)=tmp{1,ind2};Time1(i,5)=tmp{4,ind};Time1(i,6)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G2,Label,optsE);Acc2(i,3)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate({G1,G2},Label,optsE);Acc2(i,5)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Time2(i,5)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};
%         tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc3(i,1)=tmp{1,ind};Acc3(i,2)=tmp{1,ind2};Time3(i,1)=tmp{4,ind};Time3(i,4)=tmp{4,ind2};
%         tmp=GraphEncoderEvaluate(G2,Label,optsE2);Acc3(i,3)=tmp{1,ind};Acc3(i,4)=tmp{1,ind2};Time3(i,3)=tmp{4,ind};Time3(i,4)=tmp{4,ind2};
%         tmp=GraphEncoderEvaluate({G1,G2},Label,optsE2);Acc3(i,5)=tmp{1,ind};Acc3(i,6)=tmp{1,ind2};Time3(i,5)=tmp{4,ind};Time3(i,6)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder({G1,G2},Label,optsE);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3','out')
    [mean(Acc1);mean(Acc2);std(Acc1);std(Acc2);mean(Time1);mean(Time2)] %GEE; ASE; PGEE; PCA
    [sum(out.comChoice),max(Label)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==91
    tl = tiledlayout(3,3);fs=36;
     myColor = brewermap(8,'Spectral');

     for tab=1:3
    [Dis,Label]=simGenerate(290+10*tab,5000,20,0);
    Y=Label;
    Label2=[];
    for i=1:max(Label)
        ind=find(Label==i);
        Label2=[Label2;ind];
    end
    opts1=struct('Normalize',true,'Dimension',1);
    
    nexttile(tl)
    imagesc(Dis(Label2,Label2))
    colorbar
    switch tab
        case 1
           ylabel('SBM','FontSize',fs);
        case 2
            ylabel('DC-SBM','FontSize',fs);
        case 3
            ylabel('RDPG','FontSize',fs);
    end
    axis('square'); 
    if tab==1
       title('Adjacency Matrix')
    end
    set(gca,'FontSize',fs,'xtick',[],'ytick',[]);
    [ZOri]=GraphEncoder(Dis,Label);

% 
%         nexttile(tl)
%     imagesc(cov(ZOri))
%     yticks([1 10 20]);xticks([1 10 20]);
%     colorbar
%     axis('square'); 
%     if tab==1
%     title('Covariance Matrix');
%     end
%     set(gca,'FontSize',fs); 

[Z,out1]=GraphEncoder(Dis,Label,opts1);
    nexttile(tl)
    hold on
    myColor = brewermap(8,'Spectral');
    mc=repmat(myColor(7,:),20,1);
    mc(1:3,:)=repmat(myColor(2,:),3,1);
    scatter(1:20,out1.comChoice,50,mc,'filled');
%     scatter(1:20,thres1*ones(20,1),20,myColor(5,:),'filled');
    hold off
    xlim([1,20]);xticks([1 10 20]);
    if tab==1
       title('Community Score','FontSize',fs)
    end
    switch tab
        case 1
           ylim([0,8])
        case 2
            ylim([0,2.5])
        case 3
           ylim([0,5])
    end
    xlabel('Dimension','FontSize',fs)
    axis('square'); set(gca,'FontSize',fs);
     

     ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
        myColor = brewermap(4,'RdYlGn'); myColor2 = brewermap(4,'PuOr');myColor3 = brewermap(17,'Spectral');
        myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
%     nexttile(tl)
%         scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
%         scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
%         scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
%         for j=4:20
%             ind=find(Y==j);
%             scatter3(Z(ind,1), Z(ind,2),Z(ind,3),3,myColor3(j-3,:),'filled');
%         end
%         hold off
%         axis('square'); title('PCA*GEE'); set(gca,'FontSize',fs);

        nexttile(tl)
        scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
        scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
        scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
        for j=4:20
            ind=find(Y==j);
            scatter3(Z(ind,1), Z(ind,2),Z(ind,3),3,myColor3(j-3,:),'filled');
        end
        hold off
        axis('square'); 
        set(gca,'FontSize',fs); 
        if tab==1
           title('Sample Embedding'); 
           legend('Community 1','Community 2','Community 3','Location','NorthWest','FontSize',20)
        end
     end
    F.fname='FigDimension1';
    F.wh=[12 12]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

% if choice==92
%     tl = tiledlayout(3,3);fs=36;
%      myColor = brewermap(8,'Spectral');
% 
%     [Dis,Label]=simGenerate(300,5000,20,0);
%     [Z1,out1]=GraphEncoder(Dis,Label);
%     V1=cov(Z1);
%     [Dis,Label]=simGenerate(310,5000,20,0);
%     [Z2,out2]=GraphEncoder(Dis,Label);
%     V2=cov(Z2);
%     [Dis,Label]=simGenerate(320,5000,20,0);
%     [Z3,out3]=GraphEncoder(Dis,Label);
%     V3=cov(Z3);
%     
%     nexttile(tl)
%     imagesc(V1)
%     yticks([1 10 20]);xticks([1 10 20]);
%     colorbar
%     ylabel('SBM','FontSize',fs)
%     axis('square'); 
%     title('Covariance Matrix')
%     set(gca,'FontSize',fs);
%     nexttile(tl)
%     [a,b,c]=pca(Z1);
%     imagesc(abs(a));
%     yticks([1 10 20]);xticks([1 10 20]);
%     colorbar
%     title('Principal Components','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     hold on
%     plot(1:20,out1.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
% %     plot(1:20,thres1*ones(20,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
%     hold off
%     ylim([0,8]);xlim([1,20]);xticks([1 10 20]);
%     title('Importance Score','FontSize',fs)
%     xlabel('Community','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     imagesc(V2)
%     yticks([1 10 20]);xticks([1 10 20]);
%     axis('square'); 
%     ylabel('DC-SBM','FontSize',fs)
%     set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     [a,b,c]=pca(Z2);
%     imagesc(abs(a));
%     yticks([1 10 20]);xticks([1 10 20]);
%     axis('square'); set(gca,'FontSize',fs);
%     
%     nexttile(tl)
%     hold on
%     plot(1:20,out2.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
% %     plot(1:20,thres1*ones(20,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
%     ylim([0,3]);xlim([1,20]);xticks([1 10 20]);
%     hold off
%     %ylabel('Importance Score','FontSize',fs)
%     %xlabel('Community','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     imagesc(V3)
%     yticks([1 10 20]);xticks([1 10 20]);
%     axis('square'); 
%     ylabel('RDPG','FontSize',fs)
%     set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     [a,b,c]=pca(Z3);
%     imagesc(abs(a));
%     yticks([1 10 20]);xticks([1 10 20]);
%     axis('square'); set(gca,'FontSize',fs);
%     
%     nexttile(tl)
%     hold on
%     plot(1:20,out3.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
% %     plot(1:20,thres1*ones(20,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
%     ylim([0,5]);xlim([1,20]);xticks([1 10 20]);
%     hold off
%     %ylabel('Importance Score','FontSize',fs)
%     %xlabel('Community','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     F.fname='FigDimension2';
%     F.wh=[12 12]*2;
%         %     F.PaperPositionMode='auto';
%     print_fig(gcf,F)
% end
if choice==93
    tl = tiledlayout(3,3);fs=36;
    myColor = brewermap(8,'Spectral');ind=1;myColor2 = brewermap(5,'PuOr');

    for tab=1:3
    nexttile(tl)
    switch tab
        case 1
            load('GEEDimension1.mat'); titleStr='SBM';
        case 2
            load('GEEDimension2.mat'); titleStr='DC-SBM';
        case 3
            load('GEEDimension3.mat'); titleStr='RDPG';
    end
    bayes=0.25-0.25/17;

    
    hold on
    plot(1:20,Acc3(1:20,1),'Color', myColor(2,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(1:20,3),'Color', myColor(7,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,thres1*ones(20,1),'Color', myColor(5,:), 'LineStyle', '--','LineWidth',5);
    hold off
    xlim([1,20]); xticks([1 10 20]); xticklabels({'250','2500','5000'});
%     ylim([0,6.5]);
    %     xlabel('Sample Size','FontSize',fs)
    %     ylabel('SBM','FontSize',fs)
    switch tab
        case 1
           ylim([0,8])
        case 2
            ylim([0,2.5])
        case 3
           ylim([0,5])
    end
    if tab==1
        title('Community Score');legend('Principal','Redundant','Location','NorthWest')
    end
    ylabel(titleStr,'FontSize',fs)
    xlabel('Sample Size','FontSize',fs)
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    hold on
    plot(1:20,Acc3(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(:,4),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    ylim([0,1.1]);
     xlim([1,20]); xticks([1 10 20]); xticklabels({'250','2500','5000'});
%     xlabel('Sample Size','FontSize',fs)
    %ylabel('Accuracy','FontSize',fs)
    if tab==1
       title('Detection Accuracy');legend('True Positive','False Positive','Location','East');
    end
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    hold on
    plot(1:20,Acc2(:,ind),'Color', myColor2(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc1(:,ind),'Color', myColor2(4,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc4(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,bayes*ones(20,1),'Color', myColor2(2,:), 'LineStyle', ':','LineWidth',5);
%     ylim([0.2,0.8]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'250','2500','5000'});
    if tab==1
    title('Classification Error');legend('P-GEE','GEE','Bayes','Location','NorthEast')
    end
    hold off
    axis('square'); set(gca,'FontSize',fs);

%     nexttile(tl)
%     hold on
%     plot(1:20,Acc3(1:20,5),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc3(1:20,6),'Color', myColor(6,:), 'LineStyle', '-','LineWidth',5);
% %     plot(1:20,thres1*ones(20,1),'Color', myColor(3,:), 'LineStyle', '--','LineWidth',5);
%     hold off
%      xlim([1,20]); xticks([1 10 20]); xticklabels({'250','2500','5000'});
% %     ylim([0,6.5]);
% %     xlabel('Sample Size','FontSize',fs)
% %     ylabel('SBM','FontSize',fs)
%     if tab==1
%        title('Marginal Variance'); legend('Principal','Redundant','Location','NorthEast')
%     end
%     hold off
%     axis('square'); set(gca,'FontSize',fs);
    end

    F.fname='FigDimension2';
    F.wh=[12 12]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end
% 
% if choice==95
%         tl = tiledlayout(2,2);fs=36;ind=1;
%     myColor = brewermap(8,'Spectral');
%     nexttile(tl)
%     load('GEEDimension1.mat'); bayes=0.25-0.25/17;
%     hold on
%     plot(1:20,Acc1(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc2(:,ind),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc4(:,ind),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,bayes*ones(20,1),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
%     legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthEast')
%     ylim([0.1,0.8]);
%     xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
%     ylabel('SBM','FontSize',fs)
%     xlabel('Sample Size','FontSize',fs)
%     title('Classification Error')
%     hold off
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     load('GEEDimension7.mat');bayes=5:5:75; bayes=0.25-0.25./(bayes-3);lim=15;
%     hold on
%     plot(1:lim,Acc1(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,Acc2(:,ind),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,Acc4(:,ind),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,bayes,'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
%     %legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthWest')
%     ylim([0.1,0.8]);
%     xlim([1,lim]); xticks([1 (lim+1)/2 lim]); xticklabels({'5','40','75'});
%     xlabel('Dimension','FontSize',fs)
%     title('Classification Error')
%     hold off
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     load('GEEDimension2.mat');bayes=0.25-0.25/17;
%     hold on
%     plot(1:20,Acc1(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc2(:,ind),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,Acc4(:,ind),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:20,bayes*ones(20,1),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
%    % legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthEast')
%     ylim([0.15,0.6]);
%     xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
%     ylabel('DC-SBM','FontSize',fs)
%     xlabel('Sample Size','FontSize',fs)
%     %title('DC-SBM Classification')
%     hold off
%     axis('square'); set(gca,'FontSize',fs);
% 
% %     tl = tiledlayout(2,2);fs=36;
% %     myColor = brewermap(8,'Spectral');
%    
% 
%     nexttile(tl)
%     load('GEEDimension8.mat'); bayes=5:5:75; bayes=0.25-0.25./(bayes-3);
%     hold on
%     plot(1:lim,Acc1(:,ind),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,Acc2(:,ind),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,Acc4(:,ind),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
%     plot(1:lim,bayes,'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
%     %legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthWest')
%     ylim([0.15,0.6]);
%     xlim([1,lim]); xticks([1 (lim+1)/2 lim]); xticklabels({'5','40','75'});
%     xlabel('Dimension','FontSize',fs)
%     %title('DC-SBM')
%     hold off
%     axis('square'); set(gca,'FontSize',fs);
% 
%     F.fname='FigDimension3';
%     F.wh=[8 8]*2;
%         %     F.PaperPositionMode='auto';
%     print_fig(gcf,F)
% end
% if choice==35
%     tl = tiledlayout(1,4);fs=36;
%      myColor = brewermap(8,'Spectral');
% 
%     [Dis,Label]=simGenerate(302,5000,10,1);
%     [Z1,out1]=GraphEncoder(Dis,Label);
%     V1=cov(Z1);
%     [Dis,Label]=simGenerate(312,5000,10,1);
%     [Z2,out2]=GraphEncoder(Dis,Label);
%     V2=cov(Z2);
%     
%     nexttile(tl)
%     imagesc(V1)
%     ylabel('Covariance Matrix','FontSize',fs)
%     axis('square'); 
%     title('SBM')
%     set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     hold on
%     plot(1:10,out1.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
%     plot(1:10,ones(10,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
%     hold off
%     ylim([0,5]);xlim([1,10]);title('SBM')
%     ylabel('Importance Score','FontSize',fs)
%     xlabel('Dimension','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     nexttile(tl)
%     imagesc(V2)
%     axis('square'); 
%     title('DC-SBM');
%     ylabel('Covariance Matrix','FontSize',fs)
%     set(gca,'FontSize',fs);
% 
% 
%     nexttile(tl)
%     hold on
%     plot(1:10,out2.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
%     plot(1:10,ones(10,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
%     ylim([0,5]);xlim([1,10]);title('DC-SBM')
%     hold off
%     ylabel('Importance Score','FontSize',fs)
%     xlabel('Dimension','FontSize',fs)
%     axis('square'); set(gca,'FontSize',fs);
% 
%     F.fname='FigDimensionA2';
%     F.wh=[16 4]*2;
%         %     F.PaperPositionMode='auto';
%     print_fig(gcf,F)
% end

% if choice==40 %kept 20 out of 39 dimensions
%     opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30);
%     optsE = opts; optsE.Principal=1;ind=2;ind2=3;
%     load('anonymized_msft.mat')
%     Label=label;rep=1;i=1;Acc1=zeros(rep,6);Acc2=zeros(rep,6);Time1=zeros(rep,6);Time2=zeros(rep,6);
%     indices = crossvalind('Kfold',Label,5);
%     opts.indices=indices;optsE.indices=indices;
% %     tmp=GraphEncoderEvaluate(G{6},Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,4)=tmp{1,ind2};Time1(i,1)=tmp{4,ind};Time1(i,4)=tmp{4,ind2};
% %     tmp=GraphEncoderEvaluate(G{6},Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
% %     tmp=GraphEncoderEvaluate(G{12},Label,opts);Acc1(i,2)=tmp{1,ind};Acc1(i,5)=tmp{1,ind2};Time1(i,2)=tmp{4,ind};Time1(i,5)=tmp{4,ind2};
% %     tmp=GraphEncoderEvaluate(G{12},Label,optsE);Acc2(i,2)=tmp{1,ind};Acc2(i,5)=tmp{1,ind2};Time2(i,2)=tmp{4,ind};Time2(i,5)=tmp{4,ind2};
% %     tmp=GraphEncoderEvaluate(G{18},Label,opts);Acc1(i,3)=tmp{1,ind};Acc1(i,6)=tmp{1,ind2};Time1(i,3)=tmp{4,ind};Time1(i,6)=tmp{4,ind2};
% %     tmp=GraphEncoderEvaluate(G{18},Label,optsE);Acc2(i,3)=tmp{1,ind};Acc2(i,6)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,6)=tmp{4,ind2};
%     tic
%     [Z,out]=GraphEncoder(G,label,0,opts);
%     time=toc;
%     opts.Dimension=true;
%     %save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','Z','out');
%     save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','out','time');
%     [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
%     DimScore=zeros(1,39);thres=1;
%     DimChoice=(DimScore==1);
%     for i=1:24
%         DimScore=DimScore+out(i).DimScore/24;
%         DimChoice=DimChoice | (out(i).DimScore>thres);
%     end
% %     sum(out(1).DimScore>1)/length(out(1).DimScore)
% end
% 
% if choice==50
%     load('anonymized_msft.mat');
%     load('GEEDimension27.mat');tt=1;
% %     [~,Z3]=pca(Z(:,DimChoice,tt),'numComponents',3,'Centered',false);
% %     [~,Z4]=pca(Z(:,:,tt),'numComponents',3,'Centered',false);
%     [Z3,umap,clusterIdentifiers,extras]=run_umap(Z(:,DimChoice,tt),'n_components',3);
%     [Z4,umap,clusterIdentifiers,extras]=run_umap(Z(:,:,tt),'n_components',3);
%     maxK=39;
%     myColor = brewermap(maxK,'RdYlGn');
%     tl = tiledlayout(1,2);
%     nexttile(tl)
%     i=1;
%     ind=(label==i);scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
%     hold on
%     for i=2:maxK
%         ind=(label==i);
%         scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
%     end
%     title('Full GEE * UMAP')
%     hold off
%     nexttile(tl)
%     i=1;
%     ind=(label==i);scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
%     hold on
%     for i=2:maxK
%         ind=(label==i);
%         scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
%         hold on
%     end
%     hold off
%     title('Principal GEE * UMAP')
% end
% 
% if choice==60
%     load('smartphone.mat');
%     load('GEEDimension26.mat');
%     Z=GraphEncoder(Edge,Label);
%     Z2=Z(:,(out.DimScore>1));
%     [Z3,umap,clusterIdentifiers,extras]=run_umap(Z2(:,:),'n_components',3);
%     [Z4,umap,clusterIdentifiers,extras]=run_umap(Z(:,:),'n_components',3);
%      maxK=20;
%     myColor = brewermap(maxK,'RdYlGn');
%     tl = tiledlayout(1,2);
%     nexttile(tl)
%     i=1;
%     ind=(Label==i);scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
%     hold on
%     for i=2:maxK
%         ind=(Label==i);
%         scatter3(Z4(ind,1),Z4(ind,2),Z4(ind,3),'Color', myColor(i,:));
%     end
%     title('Full GEE * UMAP')
%     hold off
%     nexttile(tl)
%     i=1;
%     ind=(Label==i);scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
%     hold on
%     for i=2:maxK
%         ind=(Label==i);
%         scatter3(Z3(ind,1),Z3(ind,2),Z3(ind,3),'Color', myColor(i,:));
%         hold on
%     end
%     hold off
% %     dlmwrite('a.tsv', Z, 'delimiter', '\t');
% end

% 1. variance and importance score plot, 
% 2. FP / TP / importance score for type 1, 2，3, 9； 5，6，7 10.
% 3. classification error plot for increasing n in 1 and 5.

% 
% n=3000;k=10;type=300;
% [Dis,Label]=simGenerate(type,n,k);
% Z=GraphEncoder(Dis,Label); 
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