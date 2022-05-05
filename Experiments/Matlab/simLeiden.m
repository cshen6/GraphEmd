function simLeiden


load('email.mat')
[Z]=GraphEncoder(Adj,Y);
[X]=run_umap([Z,Y],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,Y);

rng(0);
n=3000;k=10;
[Adj,Y]=simGenerate(20,n);
[Z]=GraphEncoder(Adj,Y);
[X]=run_umap([Z,Y],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,Y);
% DCSBM0=GraphEncoderEvaluate(Adj,Y,opts2);

[Adj,Y]=simGenerate(21,n,k);
[Z]=GraphEncoder(Adj,Y);
[X]=run_umap([Z,Y],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,Y);


load('phone.mat')
% Phone=GraphEncoderEvaluate(Edge,YL);
[Z]=GraphEncoder(Edge,YL);
[X]=run_umap([Z,YL],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,Y);fs=15;
hold on
text(X(58,1),X(58,2),'Apple','FontSize',fs)
text(X(99,1),X(99,2),'Blackberry','FontSize',fs)
text(X(313,1),X(313,2),'Google','FontSize',fs)
text(X(1373,1),X(1373,2),'Samsung','FontSize',fs)
text(X(486,1),X(486,2),'Huawei','FontSize',fs)
hold off
title('2D Graph Visualization','FontSize',20)
[Idx,D] = knnsearch(Z,Z(58,:),'K',30); %%IPhoneX, 53, 
Name=phone(Idx);
Dist=D';
table(Name,Dist); 