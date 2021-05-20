function []=generatePlotAEE(opt)


fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=30;
lw=3;

if opt==1
load('simulationAEE1.mat')
figure('units','normalized','Position',[0 0 1 1]);
fs=30;
ln=1:num;
subplot(2,3,1)
hold on
% errorbar(ln,1-mean(SBM_acc_AEL,2),std(SBM_acc_AEL,0,2),'r-','LineWidth',2);
% errorbar(ln,1-mean(SBM_acc_AEK,2),std(SBM_acc_AEK,0,2),'b--','LineWidth',2);
% errorbar(ln,1-mean(SBM_acc_ASE,2),std(SBM_acc_ASE,0,2),'g.-','LineWidth',2);
plot(ln,1-mean(SBM_acc_AEE_NNE,2),'r-','LineWidth',lw);
plot(ln,1-mean(SBM_acc_AEE_NNC,2),'g-','LineWidth',lw);
plot(ln,1-mean(SBM_acc_AEE_LDA,2),'b-','LineWidth',lw);
plot(ln,1-mean(SBM_acc_ASE_NNE,2),'c--','LineWidth',lw);
plot(ln,1-mean(SBM_acc_ASE_LDA,2),'m--','LineWidth',lw);
hold off
xlim([1,num])
xticks([1,num/2,num])
% legend('AEE*NNE','AEE*NNC','AEE*LDA','ASE*NNE','ASE*LDA','Location','SouthWest');
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
ylabel('Classification Error')
title('SBM');
ylim([0,0.65])
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,4)
semilogy(ln,mean(SBM_t_AEE_NNE,2),'r-',ln,mean(SBM_t_AEE_NNC,2),'g-',ln,mean(SBM_t_AEE_LDA,2),'b-',ln,mean(SBM_t_ASE_NNE,2),'c--',ln,mean(SBM_t_ASE_LDA,2),'m--','LineWidth',lw)
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
plot(ln,1-mean(DCSBM_acc_AEE_NNE,2),'r-','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_AEE_NNC,2),'g-','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_AEE_LDA,2),'b-','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_ASE_NNE,2),'c--','LineWidth',lw);
plot(ln,1-mean(DCSBM_acc_ASE_LDA,2),'m--','LineWidth',lw);
hold off
xlim([1,num])
xticks([1,num/2,num])
legend('AEE*NNE','AEE*NNC','AEE*LDA','ASE*NNE','ASE*LDA','Location','NorthEast');
xticklabels({'100','1000','2000'})
xlabel('Sample Size')
%ylabel('Classification Error')
ylim([0,0.7])
title('DC-SBM');
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,5)
semilogy(ln,mean(DCSBM_t_AEE_NNE,2),'r-',ln,mean(DCSBM_t_AEE_NNC,2),'g-',ln,mean(DCSBM_t_AEE_LDA,2),'b-',ln,mean(DCSBM_t_ASE_NNE,2),'c--',ln,mean(DCSBM_t_ASE_LDA,2),'m--','LineWidth',lw)
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
plot(ln,1-mean(RDPG_acc_AEE_NNE,2),'r-','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_AEE_NNC,2),'g-','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_AEE_LDA,2),'b-','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_ASE_NNE,2),'c--','LineWidth',lw);
plot(ln,1-mean(RDPG_acc_ASE_LDA,2),'m--','LineWidth',lw);
hold off
xlim([1,num])
xticks([1,num/2,num])
%legend('AEE1','AEE2','ASE','Location','NorthEast');
xticklabels({'20','200','400'})
xlabel('Sample Size')
%ylabel('Classification Error')
ylim([0.2,0.7])
title('RDPG');
axis('square')
set(gca,'FontSize',fs);
subplot(2,3,6)
semilogy(ln,mean(RDPG_t_AEE_NNE,2),'r-',ln,mean(RDPG_t_AEE_NNC,2),'g-',ln,mean(RDPG_t_AEE_LDA,2),'b-',ln,mean(RDPG_t_ASE_NNE,2),'c--',ln,mean(RDPG_t_ASE_LDA,2),'m--','LineWidth',lw)
xlim([1,num])
xticks([1,num/2,num])
xticklabels({'20','200','400'})
xlabel('Sample Size')
%ylabel('Running Time (s)')
%legend('AEE1','AEE2','ASE','Location','SouthEast');
axis('square')
% ylim([0,0.8])
%title(strcat('GFN vs ASE for SBM'),'FontSize',fs)
%xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE1');
F.wh=[12 8]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end


if opt==2
% theorem 1: SBM
% n=300;fs=15;
% [Adj,Y]=simGenerate(1,n);
% [Z,filter]=GraphEncoder(Adj,Y);
% 
% % theorem 3: RDPG
% n=300;fs=15;
% [Adj,Y]=simGenerate(3,n);
% [Z,filter]=GraphEncoder(Adj,Y);

% fig 1a
n=1000;fs=15;K=2;
[Adj,Y]=simGenerate(1,n,K);
[Z,~]=GraphEncoder(Adj,Y);
figure('units','normalized','Position',[0 0 1 1]);
subplot(1,2,1)
plot3(Z(Y==1,1),Z(Y==1,2),Z(Y==1,3),'ro');
hold on
plot3(Z(Y==2,1),Z(Y==2,2),Z(Y==2,3),'gx');
plot3(Z(Y==3,1),Z(Y==3,2),Z(Y==3,3),'bs');
hold off
title('AEE Embedding for RDPG Graph','FontSize',fs)
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
% [mdl2,filter]=GraphNN(Adj,Y,0,2);

[U,S,V]=svd(Adj);d=K;
Z=U(:,1:d)*S(1:d,1:d)^0.5;
subplot(1,2,2)
plot3(Z(Y==1,1),Z(Y==1,2),Z(Y==1,3),'ro');
hold on
plot3(Z(Y==2,1),Z(Y==2,2),Z(Y==2,3),'gx');
plot3(Z(Y==3,1),Z(Y==3,2),Z(Y==3,3),'bs');
hold off
title('ASE Embedding for RDPG Graph','FontSize',fs)

F.fname=strcat('C:\Work\Applications\GitHub\GraphNN\Matlab\results\FigAEE0');
F.wh=[6 4]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end