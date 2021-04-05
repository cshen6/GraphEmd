% theorem 1: SBM clustering
n=1000;fs=15;
[Adj,Y]=generateSims(1,n,2);

tic
[ind_AEE,Z_AEE]=GraphClustering(Adj,3);
t1=toc;
figure
hold on
plot(Z_AEE(ind_AEE==1,1),Z_AEE(ind_AEE==1,2),'ro');
plot(Z_AEE(ind_AEE==2,1),Z_AEE(ind_AEE==2,2),'go');
plot(Z_AEE(ind_AEE==3,1),Z_AEE(ind_AEE==3,2),'bo');
hold off
title('AEE Clustering for SBM','FontSize',fs)
RIAEE=RandIndex(Y,ind_AEE);

tic
d=3;[U,S,V]=svds(Adj,d);
Z_ASE=U(:,1:d)*S(1:d,1:d)^0.5;
ind_ASE = kmeans(Z_ASE, 3);
t2=toc;
figure
hold on
plot(Z_ASE(ind_ASE==1,1),Z_ASE(ind_ASE==1,2),'ro');
plot(Z_ASE(ind_ASE==2,1),Z_ASE(ind_ASE==2,2),'go');
plot(Z_ASE(ind_ASE==3,1),Z_ASE(ind_ASE==3,2),'bo');
hold off
title('ASE Clustering for SBM','FontSize',fs)
RIASE=RandIndex(Y,ind_ASE);
RI=RandIndex(ind_AEE,ind_ASE);


% theorem 2: RDPG
n=1000;fs=15;
[Adj,Y]=generateSims(3,n,2);
Y=Y+1;
[Z,filter]=GraphEncoder(Adj,Y);

tic
[ind_AEE]=GraphClustering(Adj,3);
t1=toc;
[Z,filter]=GraphEncoder(Adj,ind_AEE);
figure
hold on
plot(Z_AEE(ind_AEE==1,1),Z_AEE(ind_AEE==1,2),'ro');
plot(Z_AEE(ind_AEE==2,1),Z_AEE(ind_AEE==2,2),'go');
plot(Z_AEE(ind_AEE==3,1),Z_AEE(ind_AEE==3,2),'bo');
hold off
title('AEE Clustering for RDPG','FontSize',fs)
RIAEE=RandIndex(Y,ind_AEE);

tic
d=3;[U,S,V]=svds(Adj,d);
Z_ASE=U(:,1:d)*S(1:d,1:d)^0.5;
ind_ASE = kmeans(Z_ASE, 3);
t2=toc;
figure
hold on
plot(Z_ASE(ind_ASE==1,1),Z_ASE(ind_ASE==1,2),'ro');
plot(Z_ASE(ind_ASE==2,1),Z_ASE(ind_ASE==2,2),'go');
plot(Z_ASE(ind_ASE==3,1),Z_ASE(ind_ASE==3,2),'bo');
hold off
title('ASE Clustering for RDPG','FontSize',fs)
RIASE=RandIndex(Y,ind_ASE);
RI=RandIndex(ind_AEE,ind_ASE);

% theorem 3: SBM clustering running time
n=10000;fs=15;
[Adj,Y]=generateSims(1,n,2);

tic
[ind_AEE]=GraphClustering(Adj,3);
t1=toc;
[Z,filter]=GraphEncoder(Adj,ind_AEE);
figure
hold on
plot(Z_AEE(ind_AEE==1,1),Z_AEE(ind_AEE==1,2),'ro');
plot(Z_AEE(ind_AEE==2,1),Z_AEE(ind_AEE==2,2),'go');
plot(Z_AEE(ind_AEE==3,1),Z_AEE(ind_AEE==3,2),'bo');
hold off
title('AEE Clustering for SBM','FontSize',fs)
RIAEE=RandIndex(Y,ind_AEE);

% n=100000;fs=15;
% [Adj,Y]=generateSims(1,n,2);
% 
% tic
% [ind]=GraphClustering(Adj,3);
% t1=toc;
% [Z,filter]=GraphEncoder(Adj,ind);
% figure
% hold on
% plot(Z(ind==1,1),Z(ind==1,2),'ro');
% plot(Z(ind==2,1),Z(ind==2,2),'go');
% plot(Z(ind==3,1),Z(ind==3,2),'bo');
% hold off
% title('AEE Clustering for SBM','FontSize',fs)
% RIAEE=RandIndex(Y,ind);