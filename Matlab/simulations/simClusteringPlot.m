function simClusteringPlot(type)

fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end-1));
pre=strcat(rootDir,'Matlab/results/');% The folder to save figures
fs=15;

if type==2
%     opts = struct('model','SBM','AEE',1,'ASE',1,'interval',10,'reps',1);
%     generateClustering(5000, opts)
%     opts = struct('model','RDPG','AEE',1,'ASE',1,'interval',10,'reps',1);
%     generateClustering(5000, opts)
    lw=2; s=2;t=3;
    
    filename=strcat(pre,'AEESBM10000.mat');
    load(filename);
    figure('units','normalized','position',[0 0 1 1])
    subplot(s,t,1)
    hold on
    plot(nrange,mean(RI_AEE,2),'r-','linewidth',lw);
    plot(nrange,mean(RI_ASE,2),'g-','linewidth',lw);
    legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('Stochastic Block Model');
    xlim([nrange(1),nrange(end)]);
    ylim([0,1]);
    ylabel('Adjusted Rand Index')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    subplot(s,t,4)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    hold on
    semilogy(nrange,mean(t_ASE,2),'g-','linewidth',lw);
    xlim([nrange(1),nrange(end)]);
    ylim([0.01,40]);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
%     title('Random Dot Product Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    
    filename=strcat(pre,'AEEDCSBM10000.mat');
    load(filename);
    subplot(s,t,2)
    hold on
    plot(nrange,mean(RI_AEE,2),'r-','linewidth',lw);
    plot(nrange,mean(RI_ASE,2),'g-','linewidth',lw);
    xlim([nrange(1),nrange(end)]);
    ylim([0,0.3]);
    legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('DC-SBM Model');
    ylabel('Adjusted Rand Index')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    subplot(s,t,5)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    hold on
    semilogy(nrange,mean(t_ASE,2),'g-','linewidth',lw);
    xlim([nrange(1),nrange(end)]);
    ylim([0.01,40]);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    %title('Random Dot Product Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    
    filename=strcat(pre,'AEERDPG10000.mat');
    load(filename);
    subplot(s,t,3)
    hold on
    plot(nrange,mean(RI_AEE,2),'r-','linewidth',lw);
    plot(nrange,mean(RI_ASE,2),'g-','linewidth',lw);
    xlim([nrange(1),nrange(end)]);
    ylim([0,0.6]);
    legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('Random Dot Product Model');
    ylabel('Adjusted Rand Index')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    subplot(s,t,6)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    hold on
    semilogy(nrange,mean(t_ASE,2),'g-','linewidth',lw);
    xlim([nrange(1),nrange(end)]);
    ylim([0.01,40]);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    %title('Random Dot Product Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    hold off
    
    F.fname=strcat(pre, 'Fig2');
    F.wh=[12 6]*2;
%     F.PaperPositionMode='auto'; 
    print_fig(gcf,F)
end

if type==3
%     opts = struct('model','SBM','AEE',1,'ASE',0,'interval',20,'reps',1);
%     generateClustering(20000, opts)
%     opts = struct('model','RDPG','AEE',1,'ASE',0,'interval',20,'reps',1);
%     generateClustering(20000, opts)

    filename=strcat(pre,'AEESBM20000.mat');
    load(filename);
    figure('units','normalized','position',[0 0 1 1])
    lw=2; s=1;t=3;
    subplot(s,t,1)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('Random Dot Product Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    
    filename=strcat(pre,'AEESBM20000.mat');
    load(filename);
    figure('units','normalized','position',[0 0 1 1])
    lw=2; s=1;t=3;
    subplot(s,t,1)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('Random Dot Product Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    
    filename=strcat(pre,'AEEDCSBM20000.mat');
    load(filename);
    subplot(s,t,2)
    semilogy(nrange,mean(t_AEE,2),'r-','linewidth',lw);
    %legend('AEE Clustering', 'ASE Clustering','Location','SouthEast');
    xlabel('Sample Size')
    title('DC-SBM Model');
    ylabel('Running Time')
    axis('square');
    set(gca,'FontSize',fs);
    
    F.fname=strcat(pre, 'Fig3');
    F.wh=[8 3]*2;
%     F.PaperPositionMode='auto'; 
    print_fig(gcf,F)
end