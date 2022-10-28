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
oot=struct('Laplacian',false,'Dim',0,'DiagA',true);
[Z]=GraphEncoder(Edge,YL,oot);
[X]=run_umap([Z,YL],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,Y);fs=15;
% hold on
% text(X(58,1),X(58,2),'Apple','FontSize',fs)
% text(X(99,1),X(99,2),'Blackberry','FontSize',fs)
% text(X(313,1),X(313,2),'Google','FontSize',fs)
% text(X(1373,1),X(1373,2),'Samsung','FontSize',fs)
% text(X(486,1),X(486,2),'Huawei','FontSize',fs)
% hold off
% title('Embedding for All Communities','FontSize',20)
hold on
text(X(53,1),X(53,2),'8 plus','FontSize',fs)
text(X(57,1),X(57,2),'xs','FontSize',fs)
text(X(58,1),X(58,2),'xs max','FontSize',fs)
text(X(55,1),X(55,2),'x','FontSize',fs)
text(X(52,1),X(52,2),'8','FontSize',fs)
hold off
title('Vertex Embedding for IPhones','FontSize',20)
ind=[58,99,313,486];k=20;
[n1,n2,n3]=GraphEncoderRecom(Z,YL,ind,k);
% [Idx,D] = knnsearch(Z,Z(58,:),'K',30); %%IPhoneX, 53, 
Name=phone(n1(1:4,:));
Sup=phone(find(YL==n2(1:4,:)));
sub=phone(n3(1:4,:));
% Dist=D';
% table(Name,Dist); 


%%%% Compare classification and clustering 
level=6;oot=struct('Laplacian',false,'Dim',0,'DiagA',true);oot2=struct('Laplacian',true,'Dim',0,'DiagA',true);
% [Z]=GraphEncoder(Edge,YL(:,level),oot);
Z1=[];Z2=[];ARI=zeros(level+1,2);
for i=1:level
    [tmp]=GraphEncoder(Edge,YL(:,i),oot);
    Z1=[Z1,tmp];
    ind = kmeans(tmp, K);
    ARI(i,1)=RandIndex(Y,ind);
    [tmp]=GraphEncoder(Edge,YL(:,i),oot2);
    Z2=[Z2,tmp];
    ind = kmeans(tmp, K);
    ARI(i,2)=RandIndex(Y,ind);
end
ind = kmeans(Z1, K);
ARI(level+1,1)=RandIndex(Y,ind);
ind = kmeans(Z2, K);
ARI(level+1,2)=RandIndex(Y,ind);
% netGNN = patternnet(30,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
% netGNN.trainParam.showWindow = false;
% netGNN.divideParam.trainRatio = 0.9;
% netGNN.divideParam.valRatio   = 0.1;
% netGNN.divideParam.testRatio  = 0/100;
kfold=10;acc=zeros(kfold,4);
indices=crossvalind('Kfold',Y,10);
for i = 1:kfold
    tsn = (indices == i); % tst indices
    trn = ~tsn; % trning indice
    Z=Z1;
    ZTrn=Z(trn,:);
    ZTsn=Z(tsn,:);
    YTrn=Y(trn);
    YTsn=Y(tsn);
    mdl=fitcknn(ZTrn,YTrn,'Distance','euclidean','NumNeighbors',5);
    tt=predict(mdl,ZTsn);
    acc(i,1)=acc(i,1)+mean(YTsn~=tt);
    tic
    mdl=fitcdiscr(ZTrn,YTrn,'discrimType','pseudoLinear');
    tt=predict(mdl,ZTsn);
    acc(i,2)=acc(i,2)+mean(YTsn~=tt);

    Z=Z2;
    ZTrn=Z(trn,:);
    ZTsn=Z(tsn,:);
    mdl=fitcknn(ZTrn,YTrn,'Distance','euclidean','NumNeighbors',5);
    tt=predict(mdl,ZTsn);
    acc(i,3)=acc(i,3)+mean(YTsn~=tt);
    tic
    mdl=fitcdiscr(ZTrn,YTrn,'discrimType','pseudoLinear');
    tt=predict(mdl,ZTsn);
    acc(i,4)=acc(i,4)+mean(YTsn~=tt);
end
mean(acc)
opts.indices=indices;
accOld=GraphEncoderEvaluate(Edge,Y,opts);
% mean(acc3)

