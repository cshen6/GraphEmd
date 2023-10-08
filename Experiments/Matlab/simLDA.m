function simLDA(choice,rep,spec)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<2
    rep=20;
end
if nargin<3
    spec=0;
end
ind=3;ind2=2;ind3=1;%1: Max, 2: KNN, 3: LDA.

if choice>=1 && choice<=9
    switch choice
        case 1
            [X,Label]=simGenerateDis(1,300,5);D=1-squareform(pdist(X));
        case 2
            %low-dimension where d=3, p=5,
        case 3
            %low-dimension where d=p=5
        case 4
            %high-dimension where d=3, p=100;
        case 5
            %high-dimension where d=50, p=100;
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'Normalization',true);
    Acc1=zeros(rep,5);Time1=zeros(rep,5);
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind3};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind3};
        tmp=AttributeEvaluate(X,Label,indices); Acc1(i,4)=tmp(1);Acc1(i,5)=tmp(2);Time1(i,4)=tmp(3);Time1(i,5)=tmp(4);
    end
    save(strcat('GEELDA',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Time1');
    [mean(Acc1);mean(Time1);]
    %     [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end


if choice>=10 && choice<=30
    switch choice
        case 10
            load('Coil-Rag.mat');Dist1='euclidean';D = squareform(pdist(X, Dist1)); deg = sqrt(0.5*median(D(D>0)).^2);D=exp(-D.^2/2/deg^2);
        case 11
            load('Cora.mat');Dist1='cosine';D = X*X';%?
        case 12
            load('citeseer.mat');Dist1='cosine';D = X*X';%?
        case 13
            load('Wiki_Data.mat');D=1-TE; X=ASE(D,30);
        case 14
            load('Wiki_Data.mat');D=1-TF; X=ASE(D,30);
%         case 14
%             load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%?
        case 20
            load('Cora.mat');D = edge2adj(Edge);%?
        case 21
            load('citeseer.mat');D = edge2adj(Edge);%?
        case 22
            load('email.mat');D=Adj;Label=Y; X=ASE(D,30); %?
        case 23
            load('Gene.mat');D=Adj;Label=Y; X=ASE(D,30);%?
        case 24
            load('IIP.mat');D=Adj;Label=Y; X=ASE(D,30);
        case 25
            load('LastFM.mat');D=Adj;Label=Y; X=ASE(D,30);
        case 26
            load('polblogs.mat');D=Adj;Label=Y;X=ASE(D,30);
%         case 17
%             load('protein.mat');Dist1='euclidean';D = 1-squareform(pdist(X, Dist1));%?
    end
    opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',5,'dim',30,'Normalization',true);
    Acc1=zeros(rep,5);Time1=zeros(rep,5);
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices;
       tmp=GraphEncoderEvaluate(D,Label,opts); Acc1(i,1)=tmp{1,ind};Acc1(i,2)=tmp{1,ind2};Acc1(i,3)=tmp{1,ind3};Time1(i,1)=tmp{4,ind};Time1(i,2)=tmp{4,ind2};Time1(i,3)=tmp{4,ind3};
%        tmp=AttributeEvaluate(X,Label,indices); Acc1(i,4)=tmp(1);Acc1(i,5)=tmp(2);Time1(i,4)=tmp(3);Time1(i,5)=tmp(4);
    end
    save(strcat('GEELDA',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Time1');
     [mean(Acc1);mean(Time1);]
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

if choice==101
end

if choice==102
end

if choice==103
end
