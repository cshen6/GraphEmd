function []=generatePlotAEE(opt)


fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=30;
lw=3;

opts = struct('DiagA',false,'Correlation',false,'Laplacian',false);
if opt==-1;
%     load('AEETime0.mat')
%     figure('units','normalized','Position',[0 0 1 1]);
%     fs=30;lw=3;
%     ln1=1:ln;
%     subplot(1,2,1)
%     semilogy(ln1,mean(t2,2),'-','LineWidth',lw);
%     hold on
%     semilogy(ln1,mean(t1,2),'-','LineWidth',lw);
%     semilogy(ln1,mean(t3,2),'--','LineWidth',lw);
%     semilogy(ln1,mean(t4,2),'--','LineWidth',lw);
%     hold off
%     xlim([1,ln])
%     ylim([0.5e-3,3e4]);
%     xticks([1,ln/2,ln])
%     legend('AEE (edge)','AEE (matrix)', 'ASE (sparse SVD)','GCN (30 epoch)', 'Location','NorthWest');
%     xticklabels({'10^5','10^6','2*10^6'})
%     xlabel('Number of Edges')
%     ylabel('Running Time (log scale)')
%     title('Adjacency Encoder Embedding');
%     axis('square')
%     set(gca,'FontSize',fs);
%     
%     load('AEETime1.mat')
%     subplot(1,2,2)
%     semilogy(ln1,mean(t2,2),'-','LineWidth',lw);
%     hold on
%     semilogy(ln1,mean(t1,2),'-','LineWidth',lw);
%     semilogy(ln1,mean(t3,2),'--','LineWidth',lw);
%     hold off
%     xlim([1,ln])
%     ylim([0.5e-3,3e4]);
%     xticks([1,ln/2,ln])
%     % legend('AEE*NN','AEE*NNC','AEE*LDA','ASE*NN','ASE*LDA','Location','SouthWest');
%     legend('LEE (edge)','LEE (matrix)', 'LSE (sparse SVD)', 'Location','NorthWest');
%     xticklabels({'10^5','10^6','2*10^6'})
%     xlabel('Number of Edges') %xlabel('Sample Size')
%     ylabel('Running Time (log scale)')
%     title('Laplacian Encoder Embedding');
%     axis('square')
%     set(gca,'FontSize',fs);
%     
%     F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Experiments\Matlab\FigAEE4');
%     F.wh=[8 4]*2;
%     F.PaperPositionMode='auto';
%     print_fig(gcf,F)
    
    load('AEETimeEdge.mat');
    figure('units','normalized','Position',[0 0 1 1]);
    x=[10^3,5*10^3,10^4,5*10^4,10^5,5*10^5,10^6,5*10^6,10^7,5*10^7,10^8,5*10^8,10^9];fs=25;lw=3;
%     x=1000*10.^(x-1);
    loglog(x,t1,'-','LineWidth',lw);
    hold on
    x=x(1:8);
%     semilogy(x,t2,'--','LineWidth',lw);
    loglog(x,t3,'--','LineWidth',lw);
    loglog(x,t4,'--','LineWidth',lw);
    loglog(x,t5,'--','LineWidth',lw);
    hold off
    xlim([10^3,10^9])
    xticks([10^3,10^6,10^9])
    xticklabels({'10^3','10^6','10^9'})
    legend('AEE','ASE (sparse SVD)','GCN (30 epoch)','Node2Vec','Location','NorthWest');
    xlabel('Number of Edges (log scale)')
    ylabel('Running Time (log scale)')
    axis('square')
    set(gca,'FontSize',fs);
    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'FigAEE6'));
    F.wh=[5 5]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
    %xlim([1,ln])
    %ylim([0.5e-3,3e4]);
    %xticks([1000,l000000,1000000000]);
end

if opt==0
load('simulationAEE1.mat')
figure('units','normalized','Position',[0 0 1 1]);
color=get(gca,'colororder');
myColorOrder=color;
myColorOrder(1)=color(2);
myColorOrder(2)=color(2);
myColorOrder(3)=color(1);
myColorOrder(4)=color(1);
myColorOrder(5)=color(4);
myColorOrder(6)=color(4);
fs=40;
ln=1:num;
subplot(2,3,1)
hold on
% errorbar(ln,1-mean(SBM_acc_AEL,2),std(SBM_acc_AEL,0,2),'r-','LineWidth',2);
% errorbar(ln,1-mean(SBM_acc_AEK,2),std(SBM_acc_AEK,0,2),'b--','LineWidth',2);
% errorbar(ln,1-mean(SBM_acc_ASE,2),std(SBM_acc_ASE,0,2),'g.-','LineWidth',2);
plot(ln,1-mean(SBM_acc_AEE_NN,2),'-','LineWidth',lw);
plot(ln,1-mean(SBM_acc_AEE_LDA,2),'-','LineWidth',lw);
plot(ln,1-mean(SBM_acc_ASE_NN,2),'--','LineWidth',lw);
plot(ln,1-mean(SBM_acc_ASE_LDA,2),'--','LineWidth',lw);
plot(ln,1-mean(SBM_acc_LSE_NN,2),':','LineWidth',lw);
plot(ln,1-mean(SBM_acc_LSE_LDA,2),':','LineWidth',lw);
set(gca, 'ColorOrder', myColorOrder)
hold off
xlim([1,num])
xticks([1,num/2,num])
% legend('AEE*NN','AEE*NNC','AEE*LDA','ASE*NN','ASE*LDA','Location','SouthWest');
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
ylabel('Classification Error')
title('SBM');
ylim([0.05,0.65])
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,4)
semilogy(ln,mean(SBM_t_AEE_NN,2),'-',ln,mean(SBM_t_AEE_LDA,2),'-',ln,mean(SBM_t_ASE_NN,2),'--',ln,mean(SBM_t_ASE_LDA,2),'--',ln,mean(SBM_t_LSE_NN,2),':',ln,mean(SBM_t_LSE_LDA,2),':','LineWidth',lw)
set(gca, 'ColorOrder', myColorOrder)
xlim([1,num])
xticks([1,num/2,num])
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
ylabel('Running Time (s)')
%legend('AEE1','AEE2','ASE','Location','SouthEast');
axis('square')
% ylim([0,0.8])
%title(strcat('GFN vs ASE for SBM'),'FontSize',fs)
%xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

subplot(2,3,2)
hold on
% errorbar(ln,1-mean(DCSBM_acc_AEL,2),std(DCSBM_acc_AEL,0,2),'r-','LineWidth',2);
% errorbar(ln,1-mean(DCSBM_acc_AEK,2),std(DCSBM_acc_AEK,0,2),'b--','LineWidth',2);
% errorbar(ln,1-mean(DCSBM_acc_ASE,2),std(DCSBM_acc_ASE,0,2),'g.-','LineWidth',2);
plot(ln,1-mean(DCSBM_acc_AEE_NN,2),'-','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_AEE_LDA,2),'-','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_ASE_NN,2),'--','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_ASE_LDA,2),'--','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_LSE_NN,2),':','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_LSE_LDA,2),':','LineWidth',lw);
set(gca, 'ColorOrder', myColorOrder)
hold off
xlim([1,num])
xticks([1,num/2,num])
legend('AEE*5NN','AEE*LDA','ASE*5NN','ASE*LDA','LSE*5NN','LSE*LDA','Location','NorthEast','FontSize',30);
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
%ylabel('Classification Error')
ylim([0.1,0.75])
title('DC-SBM');
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,5)
semilogy(ln,mean(DCSBM_t_AEE_NN,2),'-',ln,mean(DCSBM_t_AEE_LDA,2),'-',ln,mean(DCSBM_t_ASE_NN,2),'--',ln,mean(DCSBM_t_ASE_LDA,2),'--',ln,mean(DCSBM_t_LSE_NN,2),':',ln,mean(DCSBM_t_LSE_LDA,2),':','LineWidth',lw)
set(gca, 'ColorOrder', myColorOrder)
xlim([1,num])
xticks([1,num/2,num])
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
%ylabel('Running Time (s)')
%legend('AEE1','AEE2','ASE','Location','SouthEast');
axis('square')
% ylim([0,0.8])
%title(strcat('GFN vs ASE for SBM'),'FontSize',fs)
%xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

subplot(2,3,3)
hold on
% errorbar(ln,1-mean(RDPG_acc_AEL,2),std(RDPG_acc_AEL,0,2),'r-','LineWidth',2);
% errorbar(ln,1-mean(RDPG_acc_AEK,2),std(RDPG_acc_AEK,0,2),'b--','LineWidth',2);
% errorbar(ln,1-mean(RDPG_acc_ASE,2),std(RDPG_acc_ASE,0,2),'g.-','LineWidth',2);
plot(ln,1-mean(RDPG_acc_AEE_NN,2),'-','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_AEE_LDA,2),'-','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_ASE_NN,2),'--','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_ASE_LDA,2),'--','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_LSE_NN,2),':','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_LSE_LDA,2),':','LineWidth',lw);
set(gca, 'ColorOrder', myColorOrder)
hold off
xlim([1,num])
xticks([1,num/2,num])
%legend('AEE1','AEE2','ASE','Location','NorthEast');
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
%ylabel('Classification Error')
ylim([0.1,0.3])
title('RDPG');
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,6)
semilogy(ln,mean(RDPG_t_AEE_NN,2),'-',ln,mean(RDPG_t_AEE_LDA,2),'-',ln,mean(RDPG_t_ASE_NN,2),'--',ln,mean(RDPG_t_ASE_LDA,2),'--',ln,mean(RDPG_t_LSE_NN,2),':',ln,mean(RDPG_t_LSE_LDA,2),':','LineWidth',lw)
set(gca, 'ColorOrder', myColorOrder)
xlim([1,num])
xticks([1,num/2,num])
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
%ylabel('Running Time (s)')
%legend('AEE1','AEE2','ASE','Location','SouthEast');
axis('square')
% ylim([0,0.8])
%title(strcat('GFN vs ASE for SBM'),'FontSize',fs)
%xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

currentFolder = pwd;
F.fname=strcat(strcat(currentFolder,'FigAEE0'));
F.wh=[12 8]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end


if opt==1

n=300;fs=40;K=2;
[Adj,Y]=simGenerate(12,n,K);
figure('units','normalized','Position',[0 0 1 1]);
subplot(3,3,1)
ind1=find(Y==1);ind2=find(Y==2);
ind1=[ind1;ind2];
heatmap(Adj(ind1,ind1),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
colormap default
title('SBM')
ylabel('Adjacency Matrix')
set(gca,'FontSize',fs);

n=2000;
fs=40;K=2;
[Adj,Y]=simGenerate(12,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
subplot(3,3,4)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
crr=[0.13,0.1];
crr2=[0.1,0.13];
radius=sqrt(0.13*0.87)/sqrt(1000)*3;
ang=0:0.01:2*pi; 
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
plot(crr2(1)+xp,crr2(2)+yp,'.');
hold off
ylabel('Encoder Embedding')
set(gca,'FontSize',fs);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);

[U,S,V]=svd(Adj);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(3,3,7)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
title('','FontSize',fs)
ylabel('Spectral Embedding')
set(gca,'FontSize',fs);

n=300;
[Adj,Y]=simGenerate(22,n,K);
subplot(3,3,2)
ind1=find(Y==1);ind2=find(Y==2);
ind1=[ind1;ind2];
heatmap(Adj(ind1,ind1),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
colormap default
title('DC-SBM')
set(gca,'FontSize',fs);

n=2000;
[Adj,Y]=simGenerate(22,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
subplot(3,3,5)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
%title('DC-SBM Graph','FontSize',fs)
set(gca,'FontSize',fs);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);

[U,S,V]=svd(Adj);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(3,3,8)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
title('','FontSize',fs)
set(gca,'FontSize',fs);

n=300;
[Adj,Y]=simGenerate(32,n,K);
subplot(3,3,3)
ind1=find(Y==1);ind2=find(Y==2);
ind1=[ind1;ind2];
heatmap(Adj(ind1,ind1),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
colormap default
title('RDPG')
set(gca,'FontSize',fs);

n=2000;
[Adj,Y]=simGenerate(32,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
subplot(3,3,6)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
% title('RDPG Graph','FontSize',fs)
set(gca,'FontSize',fs);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);

[U,S,V]=svd(Adj);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(3,3,9)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
title('','FontSize',fs)
set(gca,'FontSize',fs);

currentFolder = pwd;
F.fname=strcat(strcat(currentFolder,'\FigAEE1'));
F.wh=[12 12]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end

if opt==3
    n=1000;fs=30;
    %[Adj,Y]=simGenerate(30,1000);
    load('polblogs.mat'); Y=Label+1;
    %load('lastfm.mat') %AEK K=7 %AEK K=7
    [Adj2,Y2,pval,stat]=GraphResample(Adj,Y,n);
    pval
    ind=[];ind2=[];K=max(Y);
    for i=1:K
        ind=[ind;find(Y==i)];
        ind2=[ind2;find(Y2==i)];
    end
    Adj=Adj(ind,ind);
    Adj2=Adj2(ind2,ind2);
    subplot(2,2,1)
    heatmap(Adj,'GridVisible','off');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    colorbar( 'off' )
    colormap default
    title('Blogs Network');
    ylabel('Original Adjacency');
    set(gca,'FontSize',fs);
    subplot(2,2,3)
    heatmap(Adj2,'GridVisible','off');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    colorbar( 'off' )
    colormap default
%     pval=GraphTwoSampleTest(Adj,Adj2);
    xlabel(strcat('Two-sample p-value is ',{' '}, num2str(ceil(pval*100)/100)));
    ylabel('Resampled Adjacency');
    set(gca,'FontSize',fs);
    
    load('email.mat')
    [Adj2,Y2,pval,stat]=GraphResample(Adj,Y,n);
    pval
    ind=[];ind2=[];K=max(Y);
    for i=1:K
        ind=[ind;find(Y==i)];
        ind2=[ind2;find(Y2==i)];
    end
    Adj=Adj(ind,ind);
    Adj2=Adj2(ind2,ind2);
    subplot(2,2,2)
    heatmap(Adj,'GridVisible','off');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    colorbar( 'off' )
    colormap default
    title('Email Network');
    set(gca,'FontSize',fs);
%     xlabel('Original Adjacency');
    subplot(2,2,4)
    heatmap(Adj2,'GridVisible','off');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    colorbar( 'off' )
    colormap default
    %[pval,stat]=GraphTwoSampleTest(Adj,Adj2);
    xlabel(strcat('Two-sample p-value is ',{' '}, num2str(ceil(pval*100)/100)));
    set(gca,'FontSize',fs);
%     ylabel('Resampled Adjacency');
    
    currentFolder = pwd;
F.fname=strcat(strcat(currentFolder,'FigAEE3'));
    F.wh=[8 8]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if opt==5;
        load('polblogs.mat') 
    n=size(Adj,1);fs=30;sz=12;
figure('units','normalized','Position',[0 0 1 1]);
    subplot(2,2,1)
        G = graph(Adj,'upper');
    colo=onehotencode(categorical(Y'),1);
    colo=[colo(1,:);zeros(1,n);colo(2,:)];
    plot(G,'-.dr','NodeColor',colo');
% heatmap(Adj,'GridVisible','off');
% Ax = gca;
% Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
% colorbar( 'off' )
% colormap default
title('Blogs Network');

ylabel('Graph Connection');
% axis('square')
set(gca,'FontSize',fs);
[Z,~]=GraphEncoder(Adj,Y,opts);
n2=1000;
% [Adj2,Y2]=GraphResample(Adj,Y,n2);
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
subplot(2,2,3)
plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% title('Blogs Network','FontSize',fs);
% axis('square')
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
ylabel('Encoder Embedding');
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% title('DC-SBM 1')
% ylabel('n=1000')
set(gca,'FontSize',fs);
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);

% [U,S,V]=svd(Adj);d=2;
% Z=U(:,1:d)*S(1:d,1:d)^0.5;
% subplot(3,2,5)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% title('','FontSize',fs)
% ylabel('Spectral Embedding')
% set(gca,'FontSize',fs);

load('Gene.mat')
Adj=AdjOri;Y=YOri;
    n=size(Adj,1);fs=30;
[Z,~]=GraphEncoder(Adj,Y,opts);
    subplot(2,2,2)
    G = graph(Adj,'upper');
    colo=onehotencode(categorical(Y'),1);
    colo=[colo(1,:);zeros(1,n);colo(2,:)];
    plot(G,'-.dr','NodeColor',colo');
% heatmap(Adj,'GridVisible','off');
% colormap default
% Ax = gca;
% Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
% colorbar( 'off' )
title('Gene Network')
% axis('square')
set(gca,'FontSize',fs);

D=sum(Adj,2); D=D./max(D);
dk=[mean(D(Y==1)),mean(D(Y==2))];
subplot(2,2,4)
plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% axis('square')
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
% title('Encoder Embedding','FontSize',fs)
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% title('DC-SBM 1')
% ylabel('n=1000')
set(gca,'FontSize',fs);

% [U,S,V]=svd(Adj);d=2;
% Z=U(:,1:d)*S(1:d,1:d)^0.5;
% subplot(3,2,6)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% title('','FontSize',fs)
% set(gca,'FontSize',fs);

currentFolder = pwd;
F.fname=strcat(strcat(currentFolder,'FigAEE5'));
F.wh=[8 8]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end

if opt==2;
    figure('units','normalized','Position',[0 0 1 1]);
    n=3000;fs=40;K=2;
    [Adj,Y]=simGenerate(15,n,K);
    [Z,~]=GraphEncoder(Adj,Y,opts);
    Z=Z/2;
    subplot(3,3,1)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.2,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.1,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
title('SBM','FontSize',fs)
%ylabel('SBM Graph 1')
xlim([0.05,0.25])
ylim([0.05,0.15])
% ylabel('n=1000')
ylabel('Graph 1')
set(gca,'FontSize',fs);
axis('square')

n=3000;K=2;type=16;
[Adj,Y]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
subplot(3,3,4)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.1,0.2];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.2,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
xlim([0.05,0.25])
ylim([0.05,0.25])
hold off
%title('SBM Graph 1','FontSize',fs)
% ylabel('SBM Graph 1')
ylabel('Graph 2')
set(gca,'FontSize',fs);
axis('square')

n=3000;K=2;type=17;
[Adj,Y]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
subplot(3,3,7)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.1,0.2];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.2,0.4];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
xlim([0.05,0.25])
ylim([0.1,0.5])
hold off
%title('SBM Graph 1','FontSize',fs)
ylabel('Graph 3')
xlabel('Encoder Embedding')
set(gca,'FontSize',fs);
axis('square')

n=5000;K=2;cc1=4.2;
[Adj,Y,~,theta]=simGenerate(25,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
Z=Z/0.3;
Z=Z./repmat(theta,1,2);
subplot(3,3,2)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.2,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)/0.3*cc1;
xp=radius*1.2*cos(ang);
yp=radius*0.8*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.1,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)/0.3*cc1;
xp=radius*1.2*cos(ang);
yp=radius*1*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
title('DC-SBM','FontSize',fs)
% ylabel('SBM Graph 1')
xlim([0,0.4]);
ylim([0,0.3]);
% ylabel('n=3000')
set(gca,'FontSize',fs);
axis('square')

type=26;xl=[0,0.35];yl=[0,0.4];
[Adj,Y,~,theta]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
Z=Z/0.3;
Z=Z./repmat(theta,1,2);
subplot(3,3,5)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.1,0.2];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)/0.3*cc1;
xp=radius*1*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.2,0.1];
radius=sqrt(crr(1)*(1-crr(1)))/sqrt(n/2)/0.3*cc1;
xp=radius*1.5*cos(ang);
yp=radius*0.7*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
xlim([0,0.4]);
ylim([0,0.4]);
%title('SBM Graph 1','FontSize',fs)
% title('DC-SBM 2')
set(gca,'FontSize',fs);
axis('square')

n=5000;K=2;type=27;
[Adj,Y,~,theta]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
Z=Z/0.3;
Z=Z./repmat(theta,1,2);
subplot(3,3,8)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0.1,0.2];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)/0.3*cc1;
xp=radius*1*cos(ang);
yp=radius*1.2*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[0.2,0.4];
radius=sqrt(crr(2)*(1-crr(2)))/sqrt(n/2)/0.3*cc1;
xp=radius*1.1*cos(ang);
yp=radius*1.3*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
xlim([0,0.45])
ylim([0,0.7])
%title('SBM Graph 1','FontSize',fs)
xlabel('Scaled Embedding')
set(gca,'FontSize',fs);
axis('square')

n=3000;K=2;
[Adj,Y,~,X]=simGenerate(35,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
Z=(Z-X*[0.4,0.6])./[0.4,0.6];mv=0.2;
Z(Y==1)=Z(Y==1)+mv;
subplot(3,3,3)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
crr=[0,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[mv,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
title('RDPG','FontSize',fs)
% ylabel('SBM Graph 1')
xlim([-0.1,0.3])
ylim([-0.1,0.1])
ylabel('n=2000')
set(gca,'FontSize',fs);
axis('square')

n=3000;K=2;type=36;
[Adj,Y,~,X]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
Z=(Z-X*[0.2,0.15])./[0.2,0.15];mv=0.2;
Z(Y==1)=Z(Y==1)+mv;
subplot(3,3,6)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
ang=0:0.01:2*pi; 
crr=[0,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[mv,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
hold off
xlim([-0.1,0.3])
ylim([-0.1,0.1])
%title('SBM Graph 1','FontSize',fs)
% ylabel('SBM Graph 1')
set(gca,'FontSize',fs);
axis('square')

n=3000;K=2;type=37;
[Adj,Y,~,X]=simGenerate(type,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
Z=Z/2;
Z=(Z-X*[0.15,0.2])./[0.15,0.2];mv=0.2;
Z(Y==1)=Z(Y==1)+mv;
subplot(3,3,9)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
ang=0:0.01:2*pi; 
crr=[0,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
crr=[mv,0];
radius=1/sqrt(n/2)*3;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(crr(1)+xp,crr(2)+yp,'.');
xlim([-0.1,0.3])
ylim([-0.1,0.1])
hold off
xlabel('Normalized Embedding')
%title('SBM Graph 1','FontSize',fs)
set(gca,'FontSize',fs);
axis('square')

currentFolder = pwd;
F.fname=strcat(strcat(currentFolder,'FigAEE2'));
F.wh=[12 12]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end

if opt==10

n=2000;fs=10;K=2;
[Adj,Y]=simGenerate(31,n,K);
[Z,~]=GraphEncoder(Adj,Y,opts);
subplot(2,2,1)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
ylabel('Encoder Embedding')
title('Adjacency')
set(gca,'FontSize',fs);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);
axis('square')
[U,S,V]=svd(Adj);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(2,2,3)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
title('','FontSize',fs)
ylabel('Spectral Embedding')

set(gca,'FontSize',fs);
axis('square')
opts = struct('Laplacian',1);
[Z,~]=GraphEncoder(Adj,Y,opts);
subplot(2,2,2)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
ylabel('Encoder Embedding')
title('Laplacian')
set(gca,'FontSize',fs);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);
axis('square')
D=diag(max(sum(Adj,1),1))^(-0.5);
[U,S,V]=svd(D*Adj*D);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(2,2,4)
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
% plot(Z(Y==3,1),Z(Y==3,2),'bs');
hold off
title('','FontSize',fs)
ylabel('Spectral Embedding')
set(gca,'FontSize',fs);
axis('square')
% F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE1');
F.wh=[12 12]*2;
F.PaperPositionMode='auto';
% print_fig(gcf,F)
end

% 
% if opt==2
% 
% n=200;fs=30;K=2;
% [Adj,Y]=simGenerate(15,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% figure('units','normalized','Position',[0 0 1 1]);
% subplot(3,3,1)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% xlim([0,0.3])
% ylim([0,0.2])
% title('SBM 1')
% ylabel('n=200')
% set(gca,'FontSize',fs);
% 
% n=1000;fs=30;K=2;
% [Adj,Y]=simGenerate(15,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% subplot(3,3,4)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% xlim([0,0.3])
% ylim([0,0.2])
% ylabel('n=1000')
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;
% [Adj,Y]=simGenerate(15,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% subplot(3,3,7)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% xlim([0,0.3])
% ylim([0,0.2])
% set(gca,'FontSize',fs);
% ylabel('n=5000')
% 
% n=200;fs=30;K=2;type=16;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% subplot(3,3,2)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.3])
% ylim([0,0.3])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% title('SBM 2')
% set(gca,'FontSize',fs);
% 
% n=1000;fs=30;K=2;type=16;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% subplot(3,3,5)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.3])
% ylim([0,0.3])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;type=16;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y,opts);
% subplot(3,3,8)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.3])
% ylim([0,0.3])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=200;fs=30;K=2;type=17;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% subplot(3,3,3)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.35])
% ylim([0.1,0.6])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% title('SBM 3')
% set(gca,'FontSize',fs);
% 
% n=1000;fs=30;K=2;type=17;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% subplot(3,3,6)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.35])
% ylim([0.1,0.6])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;type=17;
% [Adj,Y]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% subplot(3,3,9)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([0,0.35])
% ylim([0.1,0.6])
% hold off
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% % [mdl1,filter]=GraphNN(Adj,Y,0,1);
% % [mdl2,filter]=GraphNN(Adj,Y,0,2);
% 
% F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE2');
% F.wh=[12 12]*2;
% F.PaperPositionMode='auto';
% print_fig(gcf,F)
% end
% 
% if opt==3
% 
% n=1000;fs=30;K=2;xl=[0,0.35];yl=[0,0.3];
% [Adj,Y,~,theta]=simGenerate(25,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% figure('units','normalized','Position',[0 0 1 1]);
% subplot(3,3,1)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% title('DC-SBM 1')
% ylabel('n=1000')
% set(gca,'FontSize',fs);
% 
% n=3000;fs=30;K=2;
% [Adj,Y,~,theta]=simGenerate(25,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,4)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% ylabel('n=3000')
% set(gca,'FontSize',fs);
% 
% n=10000;fs=30;K=2;
% [Adj,Y,~,theta]=simGenerate(25,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,7)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.1,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% ylabel('n=10000')
% 
% n=1000;fs=30;K=2;type=26;xl=[0,0.35];yl=[0,0.4];
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,2)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% title('DC-SBM 2')
% set(gca,'FontSize',fs);
% 
% n=3000;fs=30;K=2;type=26;
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,5)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=10000;fs=30;K=2;type=26;
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,8)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.1];
% radius=sqrt(mean(1)*(1-mean(1)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=1000;fs=30;K=2;type=27; xl=[0,0.35];yl=[0,0.6];
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,3)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% title('DC-SBM 3')
% set(gca,'FontSize',fs);
% 
% n=3000;fs=30;K=2;type=27;
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,6)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% 
% n=10000;fs=30;K=2;type=27;
% [Adj,Y,~,theta]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% % D=sum(Adj,2); D=D./max(D);
% % dk=[mean(D(Y==1)),mean(D(Y==2))];
% % Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
% subplot(3,3,9)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0.1,0.2];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[0.2,0.4];
% radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)/0.3*3.8;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% % [mdl1,filter]=GraphNN(Adj,Y,0,1);
% % [mdl2,filter]=GraphNN(Adj,Y,0,2);
% 
% F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE3');
% F.wh=[12 12]*2;
% F.PaperPositionMode='auto';
% print_fig(gcf,F)
% end
% 
% if opt==4
% 
% n=500;fs=30;K=2;xl=[-0.2,0.4];yl=[-0.3,0.3];
% [Adj,Y,~,X]=simGenerate(35,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.4,0.6])./[0.4,0.6];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% figure('units','normalized','Position',[0 0 1 1]);
% subplot(3,3,1)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% title('RDPG 1')
% ylabel('n=500')
% set(gca,'FontSize',fs);
% 
% n=2000;fs=30;K=2;
% [Adj,Y,~,X]=simGenerate(35,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.4,0.6])./[0.4,0.6];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,4)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% ylabel('n=2000')
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;
% [Adj,Y,~,X]=simGenerate(35,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.4,0.6])./[0.4,0.6];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,7)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% ylabel('n=5000')
% 
% n=500;fs=30;K=2;type=36;
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.2,0.15])./[0.2,0.15];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,2)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% title('RDPG 2')
% set(gca,'FontSize',fs);
% 
% n=2000;fs=30;K=2;type=36;
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.2,0.15])./[0.2,0.15];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,5)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;type=36;
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.2,0.15])./[0.2,0.15];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,8)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% % ylabel('SBM Graph 1')
% set(gca,'FontSize',fs);
% 
% n=500;fs=30;K=2;type=37; 
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.15,0.2])./[0.15,0.2];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,3)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% hold off
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% %title('SBM Graph 1','FontSize',fs)
% title('RDPG 3')
% set(gca,'FontSize',fs);
% 
% n=2000;fs=30;K=2;type=37;
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.15,0.2])./[0.15,0.2];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,6)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% hold off
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% 
% n=5000;fs=30;K=2;type=37;
% [Adj,Y,~,X]=simGenerate(type,n,K);
% [Z,~]=GraphEncoder(Adj,Y);
% Z=(Z-X*[0.15,0.2])./[0.15,0.2];mv=0.2;
% Z(Y==1)=Z(Y==1)+mv;
% subplot(3,3,9)
% plot(Z(Y==1,1),Z(Y==1,2),'o');
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x');
% ang=0:0.01:2*pi; 
% mean=[0,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% mean=[mv,0];
% radius=1/sqrt(n/2)*3;
% xp=radius*cos(ang);
% yp=radius*sin(ang);
% plot(mean(1)+xp,mean(2)+yp,'.');
% xlim([xl(1),xl(2)]);
% ylim([yl(1),yl(2)]);
% hold off
% %title('SBM Graph 1','FontSize',fs)
% set(gca,'FontSize',fs);
% % [mdl1,filter]=GraphNN(Adj,Y,0,1);
% % [mdl2,filter]=GraphNN(Adj,Y,0,2);
% 
% F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE4');
% F.wh=[12 12]*2;
% F.PaperPositionMode='auto';
% print_fig(gcf,F)
% end
% 
% if opt==5
% 
%     load('polblogs.mat') 
%     n=size(Adj,1);Y=Label+1;fs=30;sz=20;
%     
% figure('units','normalized','Position',[0 0 1 1]);
%     subplot(3,2,1)
% heatmap(Adj,'GridVisible','off');
% Ax = gca;
% Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
% colorbar( 'off' )
% colormap default
% title('Blogs Network');
% ylabel('Adjacency');
% % axis('square')
% set(gca,'FontSize',fs);
% 
% [Z,~]=GraphEncoder(Adj,Y);
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% subplot(3,2,3)
% plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% % title('Blogs Network','FontSize',fs);
% % axis('square')
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% ylabel('Encoder Embedding');
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % title('DC-SBM 1')
% % ylabel('n=1000')
% set(gca,'FontSize',fs);
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% subplot(3,2,5)
% plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% ylabel('Degree Scaled Embedding')
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % title('DC-SBM 1')
% % axis('square')
% set(gca,'FontSize',fs);
% 
% load('Gene.mat')
% Adj=AdjOri;Y=YOri;
%     n=size(Adj,1);fs=30;
% [Z,~]=GraphEncoder(Adj,Y);
%     subplot(3,2,2)
% heatmap(Adj,'GridVisible','off');
% colormap default
% Ax = gca;
% Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
% Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
% colorbar( 'off' )
% title('Gene Network')
% % axis('square')
% set(gca,'FontSize',fs);
% 
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% subplot(3,2,4)
% plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% % axis('square')
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% % title('Encoder Embedding','FontSize',fs)
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % title('DC-SBM 1')
% % ylabel('n=1000')
% set(gca,'FontSize',fs);
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% subplot(3,2,6)
% plot(Z(Y==1,1),Z(Y==1,2),'o','MarkerSize',sz)
% hold on
% plot(Z(Y==2,1),Z(Y==2,2),'x','MarkerSize',sz)
% % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% hold off
% % title('Degree Scaled','FontSize',fs)
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % title('DC-SBM 1')
% % axis('square')
% set(gca,'FontSize',fs);
% 
% F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE5');
% F.wh=[12 8]*2;
% F.PaperPositionMode='auto';
% print_fig(gcf,F)
% end
% % 
% % if opt==4
% % 
% % n=500;fs=30;K=2;xl=[0,0.1];yl=[0,0.15];
% % [Adj,Y,~,theta]=simGenerate(35,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % figure('units','normalized','Position',[0 0 1 1]);
% % subplot(3,3,1)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% % ang=0:0.01:2*pi; 
% % mean=[0.04,0.06];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.06,0.09];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % %title('SBM Graph 1','FontSize',fs)
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % title('RDPG 1')
% % ylabel('n=500')
% % set(gca,'FontSize',fs);
% % 
% % n=2000;fs=30;K=2;
% % [Adj,Y,~,theta]=simGenerate(35,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,4)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% % ang=0:0.01:2*pi; 
% % mean=[0.04,0.06];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.06,0.09];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % %title('SBM Graph 1','FontSize',fs)
% % % ylabel('SBM Graph 1')
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % ylabel('n=2000')
% % set(gca,'FontSize',fs);
% % 
% % n=8000;fs=30;K=2;
% % [Adj,Y,~,theta]=simGenerate(35,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,7)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% % ang=0:0.01:2*pi; 
% % mean=[0.04,0.06];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.06,0.09];
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % % ylabel('SBM Graph 1')
% % set(gca,'FontSize',fs);
% % ylabel('n=8000')
% % 
% % n=500;fs=30;K=2;type=36;xl=[0,0.1];yl=[0,0.2];
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,2)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % % plot(Z(Y==3,1),Z(Y==3,2),'bs');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.35]*0.1;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.2;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.3;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.4;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % title('RDPG 2')
% % set(gca,'FontSize',fs);
% % 
% % n=2000;fs=30;K=2;type=36;
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,5)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.35]*0.1;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.2;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.3;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.4;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % % ylabel('SBM Graph 1')
% % set(gca,'FontSize',fs);
% % 
% % n=8000;fs=30;K=2;type=36;
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,8)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.35]*0.1;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.2;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.3;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.35]*0.4;
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % % ylabel('SBM Graph 1')
% % set(gca,'FontSize',fs);
% % 
% % n=500;fs=30;K=2;type=37; xl=[0,0.2];yl=[0,0.35];
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,3)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % 
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % title('RDPG 3')
% % set(gca,'FontSize',fs);
% % 
% % n=2000;fs=30;K=2;type=37;
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,6)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % set(gca,'FontSize',fs);
% % 
% % n=8000;fs=30;K=2;type=37;
% % [Adj,Y,~,theta]=simGenerate(type,n,K);
% % [Z,~]=GraphEncoder(Adj,Y);
% % subplot(3,3,9)
% % plot(Z(Y==1,1),Z(Y==1,2),'o');
% % hold on
% % plot(Z(Y==2,1),Z(Y==2,2),'x');
% % ang=0:0.01:2*pi; 
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.1,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.2,0.3]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.1]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.4,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % mean=[0.15,0.25;0.45,0.15]*[0.5,0.2]';
% % radius=sqrt(mean(2)*(1-mean(2)))/sqrt(n/2)*3*0.9;
% % xp=radius*cos(ang);
% % yp=radius*sin(ang);
% % plot(mean(1)+xp,mean(2)+yp,'.');
% % hold off
% % xlim([xl(1),xl(2)]);
% % ylim([yl(1),yl(2)]);
% % %title('SBM Graph 1','FontSize',fs)
% % set(gca,'FontSize',fs);
% % % [mdl1,filter]=GraphNN(Adj,Y,0,1);
% % % [mdl2,filter]=GraphNN(Adj,Y,0,2);
% % 
% % F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE4');
% % F.wh=[12 12]*2;
% % F.PaperPositionMode='auto';
% % print_fig(gcf,F)
% % end