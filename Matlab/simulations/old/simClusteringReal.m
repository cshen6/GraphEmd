function [RI_AEE,RI_AEN,RI_ASE,t_AEE,t_AEN,t_ASE,ind_AEE,ind_AEN,ind_ASE]=simClusteringReal(Adj,Y,Dist)

if nargin<3
    Dist='sqeuclidean';
end
fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end-1));
pre=strcat(rootDir,'Matlab/results/');% The folder to save figures
fs=15; plots=0; d=30;

K=length(unique(Y));
Y=Y-min(Y)+1;
opts = struct('Dist',Dist,'maxIter',50,'normalize',0); % default parameters

tic
[ind_AEN,Z_AEN]=GraphClusteringNN(Adj,K,opts);
t_AEN=toc;
RI_AEN=RandIndex(Y,ind_AEN);

tic
[ind_AEE,Z_AEE]=GraphClustering(Adj,K,opts);
t_AEE=toc;
RI_AEE=RandIndex(Y,ind_AEE);

if size(Adj,2)==2
    Edge=Adj;
    Edge=Edge-min(min(Edge))+1;
    n=size(Label,1);
    Adj=zeros(n,n);
    for i=1:n
         Adj(Edge(i,1),Edge(i,2))=1;
         Adj(Edge(i,2),Edge(i,1))=1;
    end
end

RI_ASE=zeros(d,1);t_ASE=zeros(d,1);
tic
[U,S,~]=svds(Adj,d);
t1=toc;
for j=1:d
    tic
    Z_ASE=U(:,1:j)*S(1:j,1:j)^0.5;
    ind_ASE = kmeans(Z_ASE, K);
    t_ASE(j)=t1+toc;
    RI_ASE(j)=RandIndex(Y,ind_ASE);
end
[RI_ASE,ind]=max(RI_ASE);
t_ASE=t_ASE(ind);


if plots==1
figure('units','normalized','position',[0 0 1 1])
subplot(1,3,1)
hold on
plot(Z_AEE(ind_AEE==1,1),Z_AEE(ind_AEE==1,2),'ro');
plot(Z_AEE(ind_AEE==2,1),Z_AEE(ind_AEE==2,2),'go');
plot(Z_AEE(ind_AEE==3,1),Z_AEE(ind_AEE==3,2),'bo');
plot(Z_AEE(ind_AEE==4,1),Z_AEE(ind_AEE==4,2),'kx');
plot(Z_AEE(ind_AEE==5,1),Z_AEE(ind_AEE==5,2),'cx');
hold off
title('AEE Clustering','FontSize',fs);
xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

subplot(1,3,2)
hold on
plot(Z_AEN(ind_AEN==1,1),Z_AEE(ind_AEN==1,2),'ro');
plot(Z_AEN(ind_AEN==2,1),Z_AEE(ind_AEN==2,2),'go');
plot(Z_AEN(ind_AEN==3,1),Z_AEE(ind_AEN==3,2),'bo');
plot(Z_AEN(ind_AEN==4,1),Z_AEE(ind_AEN==4,2),'kx');
plot(Z_AEN(ind_AEN==5,1),Z_AEE(ind_AEN==5,2),'cx');
hold off
title('AEN Clustering','FontSize',fs);
xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEN*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEN*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);

subplot(1,3,3)
hold on
plot(Z_ASE(ind_ASE==1,1),Z_ASE(ind_ASE==1,2),'ro');
plot(Z_ASE(ind_ASE==2,1),Z_ASE(ind_ASE==2,2),'go');
plot(Z_ASE(ind_ASE==3,1),Z_ASE(ind_ASE==3,2),'bo');
plot(Z_ASE(ind_ASE==4,1),Z_ASE(ind_ASE==4,2),'kx');
plot(Z_ASE(ind_ASE==5,1),Z_ASE(ind_ASE==5,2),'cx');
hold off
title('ASE Clustering','FontSize',fs);
xlabel(strcat('ARI = ',{' '}, num2str(round(RI_ASE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_ASE*100)/100),{' '},'seconds'));
set(gca,'FontSize',fs);


F.fname=strcat(pre, 'FigReal1');
F.wh=[12 3]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end