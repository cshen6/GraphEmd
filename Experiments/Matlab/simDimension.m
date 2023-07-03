function simDimension(choice,rep)

if nargin<2
    if choice<10
        rep=100;
    else
        rep=20;
    end
end
norma=true;
if choice==1 || choice==2 || choice ==3 || choice==4 % top 3; all; none; none; repeat for DC-SBM
    lim=20;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);dim=20; ind=2;ind2=3;
    opts = struct('Adjacency',1,'Laplacian',0,'Normalize',norma,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=true;
    optsE2=optsE; optsE2.PCA=true;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,4);Acc4=zeros(lim,6);
    type=300;
    switch choice
        case 1
            type=300;
        case 2 
            type=310;
        case 3
            type=301;
        case 4
            type=311;
    end
    for i=1:lim
        for r=1:rep
            n=300*i
            [Dis,Label]=simGenerate(type,n,dim,0);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
            G3{i,r}=GraphEncoderEvaluate(Dis,Label,optsE2);
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
            Acc4(i,1)=Acc4(i,1)+G3{i,r}{1,ind2}/rep;Acc4(i,2)=Acc4(i,2)+G3{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc4(i,4)=Acc4(i,4)+G3{i,r}{4,ind2}/rep;Acc4(i,5)=Acc4(i,5)+G3{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4','Z','out')
    [mean(Acc1);mean(Acc2);mean(Acc4)]
    out.DimScore
%     [std(Acc1);std(Acc2)]
end

if choice==7 || choice==8 || choice==9 || choice==10
    lim=20;G1=cell(lim,rep);G2=cell(lim,rep); G3=cell(lim,rep);ind=2;ind2=3;n=5000;
    switch choice
        case 7
            type=300;
        case 8 
            type=310;
        case 9
            type=301;
        case 10
            type=311;
    end
    opts = struct('Adjacency',1,'Laplacian',0,'Normalize',norma,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
    optsE = opts; optsE.Dimension=true;
    optsE2=optsE; optsE2.PCA=true;
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,4);Acc4=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            dim=5*i
            [Dis,Label]=simGenerate(type,n,dim,0);
            indices = crossvalind('Kfold',Label,10);
            opts.indices=indices;optsE.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis,Label,optsE);
            G3{i,r}=GraphEncoderEvaluate(Dis,Label,optsE2);
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
            Acc4(i,1)=Acc4(i,1)+G3{i,r}{1,ind2}/rep;Acc4(i,2)=Acc4(i,2)+G3{i,r}{1,ind}/rep;%Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,ind2}/rep;
            Acc4(i,4)=Acc4(i,4)+G3{i,r}{4,ind2}/rep;Acc4(i,5)=Acc4(i,5)+G3{i,r}{4,ind}/rep;%Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,ind2}/rep;
        end
    end
    [Z,out]=GraphEncoder(Dis,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4','Z','out')
    [mean(Acc1);mean(Acc2);mean(Acc4)]
%     out.DimScore
end

if choice>=20 && choice <30
    switch choice
        case 21
            load('CElegans.mat');G1=Ac;G2=Ag;Label=vcols; %reduce to 2/3 for G2
        case 22
            load('smartphone.mat');G1=Edge;G2=Edge;G2(:,3)=(G2(:,3)>0); %reduced to 70% for both graph
        case 23
            load('IMDB.mat');G1=Edge1;G2=Edge2;Label=Label2;
        case 24
            load('Wiki_Data.mat');G1=1-TE; G2=GE; Label=Label+1;
        case 25
           load('Cora.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
        case 26
           load('citeseer.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced 1 dim in G1
        case 27
           load('protein.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1));
        case 28
           load('COIL-RAG.mat');Dist1='cosine';G1=Edge; G2 = 1-squareform(pdist(X, Dist1)); %reduced to 88/100 dim in G2.
    end
    opts = struct('Adjacency',1,'DiagAugment',false,'Normalize',1,'Laplacian',0,'Spectral',1,'LDA',1,'GNN',0,'knn',0,'dim',30);
    optsE = opts; optsE.Dimension=1;ind=3;ind2=5;optsE.Spectral=0;
    optsE2=optsE;optsE2.PCA=true;
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
        tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc3(i,1)=tmp{1,ind};Acc3(i,2)=tmp{1,ind2};Time3(i,1)=tmp{4,ind};Time3(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G2,Label,optsE2);Acc3(i,3)=tmp{1,ind};Acc3(i,4)=tmp{1,ind2};Time3(i,3)=tmp{4,ind};Time3(i,4)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate({G1,G2},Label,optsE2);Acc3(i,5)=tmp{1,ind};Acc3(i,6)=tmp{1,ind2};Time3(i,5)=tmp{4,ind};Time3(i,6)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder({G1,G2},Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3','out')
    [mean(Acc1);mean(Acc2);mean(Acc3);std(Acc1);std(Acc2);std(Acc3);mean(Time1);mean(Time2);mean(Time3)] %GEE; ASE; PGEE; PCA
    [length(out(1).Y), length(out(1).DimScore), sum(out(1).DimScore>1)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice>=30 && choice <=35
    %30-35
    switch choice
        case 30
            load('adjnoun.mat');G1=Edge+1; Label=Y; %kept 1/2 dimension
        case 31
           load('citeseer.mat');G1=Edge; 
        case 32
            load('email.mat');G1=Edge;Label=Y; %kept 36/42 dimension
        case 33
           load('LastFM.mat');G1=Edge;Label=Y; %kept 17/18 dimension
        case 34
            load('IIP.mat');G1=Edge;Label=Y; %kept 2/3 dimension
        case 35
           load('smartphone.mat');G1=Edge; %kept 49/71 dimension
         %case 40
          %   load('Protein.mat');G1=Edge;Label=b.Y;
%         case 24
%             load('Letter.mat');G1=Edge1;Label=GraphID1;
%         case 25
%             load('COIL-RAG.mat');G1=Edge;Label=GraphID;
%         case 21
%             load('IMDB.mat');G1=Edge1;Label=GraphID1;
    end
    opts = struct('Adjacency',1,'DiagAugment',false,'Normalize',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',0,'dim',30,'Dimension',false);
    optsE = opts; optsE.Dimension=true;ind=3;ind2=2;optsE.Spectral=0;
    optsE2=optsE;optsE2.PCA=0;
    Acc1=zeros(rep,4);Acc2=zeros(rep,4);Time1=zeros(rep,4);Time2=zeros(rep,4);
%     if spec>0 && choice>5
%         G1=edge2adj(G1);G2=edge2adj(G2);
%     end
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;optsE.indices=indices;optsE2.indices=indices;
        tmp=GraphEncoderEvaluate(G1,Label,opts);Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind+2};Acc1(i,4)=tmp{1,ind2+2};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind+2};Time1(i,4)=tmp{4,ind2+2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE);Acc2(i,1)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Time2(i,1)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};
        tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc2(i,3)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder(G1,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out');
    [mean(Acc1);mean(Acc2);std(Acc1);std(Acc2);mean(Time1);mean(Time2)] %GEE; ASE; PGEE; PCA
    [length(out(1).Y), length(out(1).DimScore), sum(out(1).DimScore>1)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice>=36 && choice <51
    %30-35
    switch choice
        case 36
           load('Cora.mat');Dist1='cosine';G1=Edge; 
        case 37
            load('Gene.mat');G1=Edge;Label=Y;
        case 38
            load('IMDB.mat');G1=Edge1;Label=Label1+1;
        case 39
           load('Letter.mat');Dist1='cosine';G1=Edge1;Label=Label1;
%         case 40
%            load('polblogs.mat');G1=Edge;Label=Y;
        case 40
           load('protein.mat');Dist1='cosine';G1=Edge; 
        case 41
           load('pubmed.mat');Dist1='cosine';G1=Edge; 
%         case 40
%            load('Letter');G1=Edge1;Label=Label1;
%         case 24
%             load('Letter.mat');G1=Edge1;Label=GraphID1;
%         case 25
%             load('COIL-RAG.mat');G1=Edge;Label=GraphID;
%         case 21
%             load('IMDB.mat');G1=Edge1;Label=GraphID1;
    end
    opts = struct('Adjacency',1,'DiagAugment',false,'Normalize',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',0,'dim',30,'Dimension',false);
    optsE = opts; optsE.Dimension=true;ind=3;ind2=2;optsE.Spectral=0;
    optsE2=optsE;optsE2.PCA=true;
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
        tmp=GraphEncoderEvaluate(G1,Label,optsE2);Acc2(i,3)=tmp{1,ind};Acc2(i,4)=tmp{1,ind2};Time2(i,3)=tmp{4,ind};Time2(i,4)=tmp{4,ind2};
    end
    [Z,out]=GraphEncoder(G1,Label,0,opts);
    save(strcat('GEEDimension',num2str(choice),'.mat'),'choice','Acc1','Acc2','Time1','Time2','out');
    [mean(Acc1);mean(Acc2);std(Acc1);std(Acc2);mean(Time1);mean(Time2)] %GEE; ASE; PGEE; PCA
    [length(out(1).Y), length(out(1).DimScore), sum(out(1).DimScore>1)]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==91
    tl = tiledlayout(2,4);fs=36;
     myColor = brewermap(8,'Spectral');

    [Dis,Label]=simGenerate(300,5000,10,0);
    [Z1,out1]=GraphEncoder(Dis,Label);
    V1=cov(Z1);
    [Dis,Label]=simGenerate(310,5000,10,0);
    [Z2,out2]=GraphEncoder(Dis,Label);
    V2=cov(Z2);
    
    nexttile(tl)
    imagesc(V1)
    ylabel('SBM','FontSize',fs)
    axis('square'); 
    title('Covariance Matrix')
    set(gca,'FontSize',fs);
    nexttile(tl)
    [a,b,c]=pca(Z1);
    imagesc(abs(a));
    title('Principal Component','FontSize',fs)
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    hold on
    plot(1:10,out1.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
    plot(1:10,ones(10,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
    hold off
    ylim([0,5]);xlim([1,10]);
    title('Importance Score','FontSize',fs)
    xlabel('Community','FontSize',fs)
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension1.mat');
    hold on
    plot(3:17,Acc3(3:17,1),'Color', myColor(6,:), 'LineStyle', '-','LineWidth',5);
    plot(3:17,Acc3(3:17,3),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    hold off
    legend('Principal Communities','Common Communities','Location','NorthWest')
    xlim([3,17]); xticks([3 10 17]); xticklabels({'1000','3000','5000'});ylim([0,4]);
    xlabel('Sample Size','FontSize',fs)
    title('Importance Score')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    imagesc(V2)
    axis('square'); 
    ylabel('DC-SBM','FontSize',fs)
    set(gca,'FontSize',fs);

    nexttile(tl)
    [a,b,c]=pca(Z2);
    imagesc(abs(a));
    axis('square'); set(gca,'FontSize',fs);
    
    nexttile(tl)
    hold on
    plot(1:10,out2.DimScore,'x','Color', myColor(2,:), 'LineStyle', 'none','LineWidth',5,'MarkerSize',25);
    plot(1:10,ones(10,1),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',5);
    ylim([0,5]);xlim([1,10]);
    hold off
    %ylabel('Importance Score','FontSize',fs)
    xlabel('Community','FontSize',fs)
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension2.mat');
    hold on
    plot(3:17,Acc3(3:17,1),'Color', myColor(6,:), 'LineStyle', '-','LineWidth',5);
    plot(3:17,Acc3(3:17,3),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    hold off
    %legend('Principal Communities','Common Communities','Location','East')
    ylim([0,4]);
    xlim([3,17]); xticks([3 10 17]); xticklabels({'1000','3000','5000'});
    xlabel('Sample Size','FontSize',fs)
    hold off
    axis('square'); set(gca,'FontSize',fs);

    F.fname='FigDimension1';
    F.wh=[16 8]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end
if choice==92
    tl = tiledlayout(2,4);fs=36;
    myColor = brewermap(8,'Spectral');
    nexttile(tl)
    load('GEEDimension1.mat'); bayes=0.25-0.25/17;
    hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,bayes*ones(20,1),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthEast')
    ylim([0.1,0.8]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    ylabel('SBM','FontSize',fs)
    xlabel('Sample Size','FontSize',fs)
    title('Classification Error')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension1.mat');
    hold on
    plot(1:20,Acc3(:,2),'Color', myColor(7,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(:,4),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    legend('True Positive','False Positive','Location','East')
    ylim([0,1]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    xlabel('Sample Size','FontSize',fs)
    %ylabel('Accuracy','FontSize',fs)
    title('Detection Accuracy')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension7.mat');bayes=5:5:100; bayes=0.25-0.25./(bayes-3);
    hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,bayes,'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthWest')
    ylim([0.1,0.8]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    xlabel('Dimension','FontSize',fs)
    title('Classification Error')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension7.mat');
    hold on
    plot(1:20,Acc3(:,2),'Color', myColor(7,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(:,4),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    legend('True Positive','False Positive','Location','West')
    ylim([0,1]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    xlabel('Dimension','FontSize',fs)
    title('Detection Accuracy')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension2.mat');bayes=0.25-0.25/17;
    hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,bayes*ones(20,1),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthEast')
    ylim([0.15,0.6]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    ylabel('DC-SBM','FontSize',fs)
    xlabel('Sample Size','FontSize',fs)
    %title('DC-SBM Classification')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension2.mat');
    hold on
    plot(1:20,Acc3(:,2),'Color', myColor(7,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(:,4),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    legend('True Positive','False Positive','Location','East')
    ylim([0,1]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    xlabel('Sample Size','FontSize',fs)
    %ylabel('Accuracy','FontSize',fs)
    hold off
    %title('DC-SBM Principal Community')
    axis('square'); set(gca,'FontSize',fs);

%     tl = tiledlayout(2,2);fs=36;
%     myColor = brewermap(8,'Spectral');
   

    nexttile(tl)
    load('GEEDimension8.mat'); bayes=5:5:100; bayes=0.25-0.25./(bayes-3);
    hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,bayes,'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    legend('GEE','P-GEE','PCA*GEE','Bayes','Location','NorthWest')
    ylim([0.15,0.6]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    xlabel('Dimension','FontSize',fs)
    %title('DC-SBM')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEEDimension8.mat');
    hold on
    plot(1:20,Acc3(:,2),'Color', myColor(7,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc3(:,4),'Color', myColor(3,:), 'LineStyle', '-','LineWidth',5);
    legend('True Positive','False Positive','Location','West')
    ylim([0,1]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    xlabel('Dimension','FontSize',fs)
    %title('DC-SBM')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    F.fname='FigDimension2';
    F.wh=[16 8]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if choice==93
    tl = tiledlayout(1,4);fs=36;
     myColor = brewermap(8,'Spectral');

%     [Dis,Label]=simGenerate(301,5000,10,1);
%     [Z1,out1]=GraphEncoder(Dis,Label);
%     V1=cov(Z1);
%     [Dis,Label]=simGenerate(311,5000,10,1);
%     [Z2,out2]=GraphEncoder(Dis,Label);
%     V2=cov(Z2);
%     
nexttile(tl)
   load('GEEDimension3.mat');
   hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    legend('GEE','P-GEE','PCA*GEE','Location','North')
    ylim([0,0.8]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    ylabel('Classification Error','FontSize',fs)
    xlabel('Sample Size','FontSize',fs)
    title('SBM');
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
   load('GEEDimension9.mat');
    hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    ylim([0,0.8]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    %ylabel('SBM','FontSize',fs)
    xlabel('Dimension','FontSize',fs)
    title('SBM');
    %title('Classification Error')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    nexttile(tl)
   load('GEEDimension4.mat');
 hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
%     legend('GEE','P-GEE','PCA*GEE','Location','SouthEast')
    ylim([0,0.6]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'300','3000','6000'});
    %ylabel('DC-SBM','FontSize',fs)
    xlabel('Sample Size','FontSize',fs)
    title('DC-SBM');
    hold off
    axis('square'); set(gca,'FontSize',fs);
   
   nexttile(tl)
   load('GEEDimension10.mat');
     hold on
    plot(1:20,Acc1(:,2),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc2(:,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);
    plot(1:20,Acc4(:,2),'Color', myColor(4,:), 'LineStyle', '-','LineWidth',5);
    ylim([0,0.6]);
    xlim([1,20]); xticks([1 10 20]); xticklabels({'5','50','100'});
    %ylabel('SBM','FontSize',fs)
    xlabel('Dimension','FontSize',fs)
    title('DC-SBM');
    %title('Classification Error')
    hold off
    axis('square'); set(gca,'FontSize',fs);

    F.fname='FigDimension3';
    F.wh=[16 4]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end
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
%     optsE = opts; optsE.Dimension=1;ind=2;ind2=3;
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