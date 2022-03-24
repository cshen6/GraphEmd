function []=generatePlotModel()

n=3000;K=3;
[Adj,Y]=simGenerate(11,n,K); %SBM: 10, 11, 12, 15,16,17; DC: +1
[pi,B,theta,Z]=GraphSBM(Adj,Y);
figure('units','normalized','Position',[0 0 1 1]);
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');

[pval,stat]=GraphTwoSampleTest(Adj,Adj2)

% plot(Z(Y==3,1),Z(Y==3,2),'bs');
ang=0:0.01:2*pi; 
mea=[0.2,0.1];
radius=sqrt(mea(1)*(1-mea(1)))/sqrt(n/2)/0.3*3.8;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(mea(1)+xp,mea(2)+yp,'.');
mea=[0.1,0.1];
radius=sqrt(mea(1)*(1-mea(1)))/sqrt(n/2)/0.3*3.8;
xp=radius*cos(ang);
yp=radius*sin(ang);
plot(mea(1)+xp,mea(2)+yp,'.');
hold off
%title('SBM Graph 1','FontSize',fs)
xlim([xl(1),xl(2)]);
ylim([yl(1),yl(2)]);
title('DC-SBM 1')
ylabel('n=1000')
set(gca,'FontSize',fs);