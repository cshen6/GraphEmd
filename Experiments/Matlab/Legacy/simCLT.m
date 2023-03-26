function []=simCLT(opt)

% Visualization
fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=15;
lw=3;
rep=100;
opts0 = struct('DiagA',true,'Normalize',false,'Laplacian',false,'Replicates',1);
opts1 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',1);
opts2 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',10);
% map2 = brewermap(128,'PiYG'); % brewmap
% colormap(gca,map2);

if opt==0
    n=5000;
    [Adj,Y]=simGenerate(27,n);
    K=2;
    Y2=randi([1,K],[n,1]);
    Z1=GraphEncoder(Adj,Y2);

    subplot(1,3,1);
    hold on
    plot(Z1(Y2==1,1),Z1(Y2==1,2),'o');
    plot(Z1(Y2==2,1),Z1(Y2==2,2),'x');
    plot(Z1(Y2==3,1),Z1(Y2==3,2),'*');
    title('Random Init');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y2)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    Y3 = kmeans(Z1, K);
    Z2=GraphEncoder(Adj,Y3);
    for i=1:4
        Y3 = kmeans(Z2, K);
        Z2=GraphEncoder(Adj,Y3);
    end

    Z2=GraphEncoder(Adj,Y3);
    subplot(1,3,2);
    hold on
    plot(Z2(Y3==1,1),Z2(Y3==1,2),'o');
    plot(Z2(Y3==2,1),Z2(Y3==2,2),'x');
    plot(Z2(Y3==3,1),Z2(Y3==3,2),'*');
    title('Iteration 4');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y3)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    for i=1:4
        Y3 = kmeans(Z2, K);
        Z2=GraphEncoder(Adj,Y3);
    end
    subplot(1,3,3);
    hold on
    plot(Z2(Y3==1,1),Z2(Y3==1,2),'o');
    plot(Z2(Y3==2,1),Z2(Y3==2,2),'x');
    plot(Z2(Y3==3,1),Z2(Y3==3,2),'*');
    title('Iteration 8');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y3)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'\FigCLT0'));
    F.wh=[6 2]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if opt==1
    n=5000;K=5;
    [Adj,Y]=simGenerate(21,n,K);
    Y2=randi([1,K],[n,1]);
    Z1=GraphEncoder(Adj,Y2);

    subplot(1,3,1);
    for i=1:K
       plot3(Z1(Y2==i,1),Z1(Y2==i,2),Z1(Y2==i,3),'x');
       hold on
    end
    title('Random Init');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y2)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    Y3 = kmeans(Z1, K);
    Z2=GraphEncoder(Adj,Y3);
    for i=1:2
        Y3 = kmeans(Z2, K);
        Z2=GraphEncoder(Adj,Y3);
    end

    Z2=GraphEncoder(Adj,Y3);
    subplot(1,3,2);
    for i=1:K
       plot3(Z2(Y3==i,1),Z2(Y3==i,2),Z2(Y3==i,3),'x');
       hold on
    end
    title('Iteration 3');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y3)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    for i=1:2
        Y3 = kmeans(Z2, K);
        Z2=GraphEncoder(Adj,Y3);
    end
    subplot(1,3,3);
    for i=1:K
       plot3(Z2(Y3==i,1),Z2(Y3==i,2),Z2(Y3==i,3),'x');
       hold on
    end
    title('Iteration 5');
    xlabel(strcat('ARI =',{' '},num2str(floor(RandIndex(Y,Y3)*100)/100)));
    hold off
    axis('square')
    set(gca,'xtick',[],'ytick',[]);
    set(gca,'FontSize',fs);

    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'\FigCLT1'));
    F.wh=[6 2]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)

    n=5000;K=5;
    [Adj,Y]=simGenerate(27,n,K);opts.Normalize=false;
    GraphClusteringEvaluate(Adj,Y,opts)
end

if opt==2
    n=5000;
    % opts = struct('ASE',1,'LSE',1,'NN',0,'Dist','cosine','maxIter',20,'normalize',0,'deg',0,'dmax',30); % default parameters
    K=10;rep=10; tt=3;
    SBM=zeros(3,5,tt);
    RDPG=zeros(3,5,tt);
    DCSBM=zeros(3,5,tt);
    for r=1:rep
        [Adj,Y]=simGenerate(21,n,5);
        % Edge=Adj2Edge(Adj);
        DCSBM(:,:,2)=DCSBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
        [Adj,Y]=simGenerate(27,n,K);
        % Edge=Adj2Edge(Adj);
        DCSBM(:,:,3)=DCSBM(:,:,3)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% 
%         n=5000;K=5;
%         [Adj,Y]=simGenerate(21,n,K);opts.Normalize=false;
%         GraphClusteringEvaluate(Adj,Y,opts)
    end

%     subplot(1,2,1);
% plot(2:kmax,1-score(2:kmax),'r-','LineWidth',2)
% title(strcat('1-MeanSS at K=',num2str(K)))
% xlim([2,kmax]);
% axis('square')
% set(gca,'FontSize',15);
% subplot(1,2,2);
% plot(2:kmax,ari(2:kmax),'b-','LineWidth',2)
% title(strcat('ARI at K=',num2str(K)))
% xlim([2,kmax]);
% axis('square')
% set(gca,'FontSize',15);
end