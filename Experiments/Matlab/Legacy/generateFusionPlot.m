function generateFusionPlot

%%% visualization figure fig4
fs=30;
figure('units','normalized','Position',[0 0 1 1]);
[Dis,Label]=simGenerate(11,1000,1);
ind1=find(Label==2);
ind2=find(Label==1);
ind3=find(Label==3);
ind4=find(Label==4);
ind=[ind1;ind2;ind3;ind4];
Dis=Dis(ind,ind,:); Label=Label(ind);
subplot(1,3,1)
heatmap(Dis(:,:,1),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
title('Graph 1 from SBM(B1)')
set(gca,'FontSize',fs);
subplot(1,3,2)
heatmap(Dis(:,:,2),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
title('Graph 2 from SBM(B2)')
set(gca,'FontSize',fs);
subplot(1,3,3)
heatmap(Dis(:,:,3),'GridVisible','off');
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
colorbar( 'off' )
title('Graph 3 from SBM(B3)')
set(gca,'FontSize',fs);

%%% result figure fig5
load('SBM_Fusion.mat')
fs=13;
ln=1:10;
hold on
plot(ln,G13_error_GFN,'g:',ln,G12_error_GFN,'g--',ln,G11_error_GFN,'g-.',ln,G23_error_GFN,'b:',ln,G21_error_GFN,'b--',ln,G23_error_GFN,'b-.',ln,G3_error_GFN,'r-','LineWidth',2)
hold off
xlim([1,10])
xticks([1,5,10])
xticklabels({'100','500','1000'})
ylim([0,0.8])
legend('Graph 1','Graph 2','Graph 3', 'Graph 1+2', 'Graph 2+3', 'Graph 1+3', 'All Graphs', 'FontSize',fs,'Location','eastoutside')
xlabel('Sample Size','FontSize',fs)
ylabel('Classification Error','FontSize',fs)
title('Graph Fusion Neural Network on SBMs','FontSize',fs)
axis('square')