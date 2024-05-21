function simDistance(choice,rep,spec)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE
% simLDA(10,50)

rng("default")
if nargin<2
    rep=2;
end
if nargin<3
    spec=0;
end
ind=3;ind2=2;ind3=1;%1: Max, 2: KNN, 3: LDA.

if choice>=1 && choice<=5
    type=choice;
    switch choice
%         case 1
%             Dist1='Euclidean';nn=20; %D=X*X' & false normalization can also work
%         case 2
%             Dist1='Euclidean';nn=20; %D=X*X' & false normalization can also work
%         case 3
%             Dist1='Euclidean';nn=100; %D=X*X' & false normalization can also work
        case 1
            Dist1='Euclidean';nn=50; type=101;d=100;%D=X*X' & false normalization can also work
        case 2
            Dist1='Euclidean';nn=50; type=100;%D=X*X' & false normalization can also work
    end
    lim=10;
    Acc1=zeros(lim,rep,5);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'Normalize',true);
    for i=1:lim
        for r=1:rep
            n=nn*i
            if type<100
                [X,Label]=simGenerateDis(type,n,3,d); D=squareform(pdist(X, Dist1));D=max(max(D))-D;
            else
                [D,Label]=simGenerateDis(type,n,3); X=ASE(D,30);
            end
%             D=X*X';
            indices = crossvalind('Kfold',Label,5);
            opts.indices=indices;
            tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,r,1)=tmp{1,ind};Acc1(i,r,2)=tmp{1,ind2};Acc1(i,r,3)=tmp{1,ind3};
            tmp=AttributeEvaluate(X,Label,indices); Acc1(i,r,4)=tmp(1);Acc1(i,r,5)=tmp(2);
        end
    end
    save(strcat('GEELDASim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1');
    [mean(Acc1(:,:,1),2),mean(Acc1(:,:,2),2),mean(Acc1(:,:,3),2),mean(Acc1(:,:,4),2),mean(Acc1(:,:,5),2);]
    %     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==8
    tl = tiledlayout(2,2);
    opts = struct('Normalize',true,'Discriminant',false);fs=30;Dist1='Euclidean';
    [X,Label]=simGenerateDis(101,1000,4,100);
    D=squareform(pdist(X, Dist1));D=max(max(D))-D;
    Z=GraphEncoder(D,Label,opts);Y=Label;

    ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);ind4=find(Y==4);ind5=find(Y==5);
    myColor = brewermap(15,'Spectral');
    myColor=[myColor(1,:);myColor(5,:);myColor(11,:);myColor(15,:);myColor(9,:)];
    nexttile(tl)
    scatter3(X(ind1,1), X(ind1,2),X(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(X(ind2,1), X(ind2,2),X(ind2,3),20,myColor(2,:),'filled');
    scatter3(X(ind3,1), X(ind3,2),X(ind3,3),20,myColor(3,:),'filled');
    scatter3(X(ind4,1), X(ind4,2),X(ind4,3),20,myColor(4,:),'filled');
    scatter3(X(ind5,1), X(ind5,2),X(ind5,3),20,myColor(5,:),'filled');
    hold off
    axis('square'); title('HD Gaussian in 3D (K=4)');
    set(gca,'FontSize',fs);

    % [X,Label]=simGenerateDis(101,1000,3,100);
    D=squareform(pdist(X, Dist1));D=max(max(D))-D;
%     D=X*X';
    Z=GraphEncoder(D,Label,opts);Y=Label;
    % ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    nexttile(tl)
    scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
    scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
    scatter3(Z(ind4,1), Z(ind4,2),Z(ind4,3),20,myColor(4,:),'filled');
    scatter3(Z(ind5,1), Z(ind5,2),Z(ind5,3),20,myColor(5,:),'filled');
    hold off
    axis('square'); title('Euclidean Distance * Encoder');
    set(gca,'FontSize',fs);

    % [X,Label]=simGenerateDis(101,1000,3,100);
    D=squareform(pdist(X, 'Spearman'));D=max(max(D))-D;
%     D=X*X';
    Z=GraphEncoder(D,Label,opts);Y=Label;
    % ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    nexttile(tl)
    scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
    scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
    scatter3(Z(ind4,1), Z(ind4,2),Z(ind4,3),20,myColor(4,:),'filled');
    scatter3(Z(ind5,1), Z(ind5,2),Z(ind5,3),20,myColor(5,:),'filled');
    hold off
    axis('square'); title('Rank Correlation * Encoder');
    set(gca,'FontSize',fs);

    % [X,Label]=simGenerateDis(101,1000,3,100);
    D=X*X';
%     D=X*X';
    Z=GraphEncoder(D,Label,opts);Y=Label;
    % ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    nexttile(tl)
    scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
    scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
    scatter3(Z(ind4,1), Z(ind4,2),Z(ind4,3),20,myColor(4,:),'filled');
    scatter3(Z(ind5,1), Z(ind5,2),Z(ind5,3),20,myColor(5,:),'filled');
    hold off
    axis('square'); title('Inner Product * Encoder');
    set(gca,'FontSize',fs);

    F.fname='FigLDA2';
    F.wh=[8 8]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if choice==9
    tl = tiledlayout(3,2);
    opts = struct('Normalize',true,'Discriminant',false);fs=32;Dist1='Euclidean';
    [X,Label]=simGenerateDis(101,1000,3,100);
    D=squareform(pdist(X, Dist1));D=max(max(D))-D;
    Z=GraphEncoder(D,Label,opts);Y=Label;

    ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    myColor = brewermap(4,'RdYlGn'); myColor2 = brewermap(4,'PuOr');myColor3 = brewermap(17,'Spectral');
    myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
    nexttile(tl)
    scatter3(X(ind1,1), X(ind1,2),X(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(X(ind2,1), X(ind2,2),X(ind2,3),20,myColor(2,:),'filled');
    scatter3(X(ind3,1), X(ind3,2),X(ind3,3),20,myColor(3,:),'filled');
    hold off
    axis('square'); title('HD Gaussian in 3D');
    set(gca,'FontSize',fs);
    
    [X,Label]=simGenerateDis(100,1000,3); D=X;Y=Label;
    ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);ind=[ind1;ind2;ind3];
    nexttile(tl)
    imagesc(D(ind,ind));
    colorbar( 'off' )
    axis('square'); title('Weight Graph Heatmap');
    set(gca,'FontSize',fs,'XTick',[],'YTick',[]);

    [X,Label]=simGenerateDis(101,1000,3,100);
    D=squareform(pdist(X, Dist1));D=max(max(D))-D;
%     D=X*X';
    Z=GraphEncoder(D,Label,opts);Y=Label;
    ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    nexttile(tl)
    scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
    scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
    hold off
    axis('square'); title('Distance * Encoder');
    set(gca,'FontSize',fs);

    [X,Label]=simGenerateDis(100,1000,3); D=X;
    Z=GraphEncoder(D,Label,opts);Y=Label;
    ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
    nexttile(tl)
    scatter3(Z(ind1,1), Z(ind1,2),Z(ind1,3),20,myColor(1,:),'filled');hold on
    scatter3(Z(ind2,1), Z(ind2,2),Z(ind2,3),20,myColor(2,:),'filled');
    scatter3(Z(ind3,1), Z(ind3,2),Z(ind3,3),20,myColor(3,:),'filled');
    hold off
    axis('square'); title('Adjacency * Encoder');
    set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEELDASim1Spec0.mat'); i=10;myColor2 = brewermap(11,'RdYlGn'); lw=4;
    AccT=[mean(Acc1(:,:,1),2),mean(Acc1(:,:,2),2),mean(Acc1(:,:,3),2),mean(Acc1(:,:,4),2),mean(Acc1(:,:,5),2);];
    plot(1:i,AccT(:,1),'Color', myColor2(1,:), 'LineStyle', ':','LineWidth',lw);hold on
    plot(1:i,AccT(:,2),'Color', myColor2(4,:), 'LineStyle', '-.','LineWidth',lw);
    plot(1:i,AccT(:,4),'Color', myColor2(7,:), 'LineStyle', '-','LineWidth',lw);
    plot(1:i,AccT(:,5),'Color', myColor2(10,:), 'LineStyle', '--','LineWidth',lw);
    hold off
    xlabel('Sample Size')
    ylabel('Error')
    axis('square'); xlim([1,10]);ylim([0.15,0.7]);xticks([1 5 10]);xticklabels({'50','250','500'});title('5-Fold Error');
    legend('Encoder*LDA','Encoder*kNN','LDA','kNN','Location','NorthEast');
    set(gca,'FontSize',fs);

    nexttile(tl)
    load('GEELDASim2Spec0.mat'); 
    AccT=[mean(Acc1(:,:,1),2),mean(Acc1(:,:,2),2),mean(Acc1(:,:,3),2),mean(Acc1(:,:,4),2),mean(Acc1(:,:,5),2);];
    plot(1:i,AccT(:,1),'Color', myColor2(1,:), 'LineStyle', ':','LineWidth',lw);hold on
    plot(1:i,AccT(:,2),'Color', myColor2(4,:), 'LineStyle', '-.','LineWidth',lw);
    plot(1:i,AccT(:,4),'Color', myColor2(7,:), 'LineStyle', '-','LineWidth',lw);
    plot(1:i,AccT(:,5),'Color', myColor2(10,:), 'LineStyle', '--','LineWidth',lw);
    hold off
    xlabel('Sample Size')
    ylabel('Error')
    axis('square'); xlim([1,10]);ylim([0,0.7]);xticks([1 5 10]);xticklabels({'50','250','500'});title('5-Fold Error');
    legend('Encoder*LDA','Encoder*kNN','ASE*LDA','ASE*kNN','Location','NorthEast');
    set(gca,'FontSize',fs);

    F.fname='FigLDA1';
    F.wh=[8 12]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if choice>=10 && choice<=30
    switch choice
%         case 10
%             load('20Newsgroups.mat');X=fea;Label=gnd; D=X*X';
%         case 11
%             load('20NewsHome.mat');X=fea;Label=gnd; D=X*X';
        case 10
            load('Cora.mat');Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
        case 11
            load('citeseer.mat');Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
        case 12
            load('Isolet.mat');X=fea;Label=gnd; Dist1='Euclidean';D = 1-squareform(pdist(X, Dist1));%?
        case 13
            load('COIL20.mat');X=fea;Label=gnd;Dist1='Euclidean';D = 1-squareform(pdist(X, Dist1));%?
        case 14
           load("faceYaleB_32x32"); X=fea;Label=gnd;Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
        % case 15
        %    load('faceORL_32x32.mat');X=fea;Label=gnd;Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
        case 15
            load("facePIE_32x32"); X=fea;Label=gnd;Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
        case 16
            load('Wiki_Data.mat');D=1-TE; X=ASE(D,30);
        case 17
            load('Wiki_Data.mat');D=1-TF; X=ASE(D,30);
%         case 17
%             load('Wiki_Data.mat');D=GE; X=ASE(D,30);
%         case 18
%             load('Wiki_Data.mat');D=GF; X=ASE(D,30);
%         case 13
%             load('RCV1_4Class.mat');X=fea;Label=gnd; D=X*X';
%         case 14
%             load('Reuters21578.mat');X=fea;Label=gnd; D=X*X';
%         case 15
%             load('TDT2.mat');X=fea;Label=gnd; D=X*X';
%         case 20
%             load('COIL100.mat'); X=fea;Label=gnd;Dist1='spearman';D = 1-squareform(pdist(X, Dist1));%?
%         case 14
%             load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%?
%         case 20
%             load('Cora.mat');D = edge2adj(Edge);%?
%         case 21
%             load('citeseer.mat');D = edge2adj(Edge);%?
%         case 22
%             load('email.mat');D=Adj;Label=Y; X=ASE(D,30); %?
%         case 23
%             load('Gene.mat');D=Adj;Label=Y; X=ASE(D,30);%?
%         case 24
%             load('IIP.mat');D=Adj;Label=Y; X=ASE(D,30);
%         case 25
%             load('LastFM.mat');D=Adj;Label=Y; X=ASE(D,30);
%         case 26
%             load('polblogs.mat');D=Adj;Label=Y;X=ASE(D,30);
%         case 17
%             load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%?
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',0,'knn',5,'dim',30,'Normalization',true);
    Acc1=zeros(rep,6);Time1=zeros(rep,6);
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices;
       tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind3};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind3};
       tmp=AttributeEvaluate(X,Label,indices,1); Acc1(i,4)=tmp(1,1);Time1(i,4)=tmp(2,1);
       tmp=AttributeEvaluate(X,Label,indices,2); Acc1(i,5)=tmp(1,1);Time1(i,5)=tmp(2,1);
       tmp=AttributeEvaluate(X,Label,indices,3); Acc1(i,6)=tmp(1,1);Time1(i,6)=tmp(2,1);
    end
    save(strcat('GEELDA',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Time1','rep');
     [mean(Acc1);mean(Time1);std(Acc1);std(Time1);]
%     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end
