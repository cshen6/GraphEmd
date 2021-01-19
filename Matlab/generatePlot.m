function []=generatePlot

% fig 1a
n=300;fs=15;
[Adj,Y]=generateSims(2,n,2);
% [mdl1,filter]=GraphNN(Adj,Y,0,1);
[mdl2,filter]=GraphNN(Adj,Y,0,2);

[U,S,V]=svd(Adj);d=2;
X0=U(:,1:d)*S(1:d,1:d)^0.5;
figure
hold on
plot(X0(Y==0,1),X0(Y==0,2),'ro');
plot(X0(Y==1,1),X0(Y==1,2),'bo');
hold off
title('ASE Embedding for RDPG Graph','FontSize',fs)

% X1=Adj*mdl1.IW{1,1}'*mdl1.LW{2,1}';
% figure
% hold on
% plot(X1(Y==0,1),X1(Y==0,2),'ro');
% plot(X1(Y==1,1),X1(Y==1,2),'bo');
% hold off
% title('GNN Embedding for RDPG Graph','FontSize',fs)
X2=Adj*filter*mdl2.IW{1,1}'*mdl2.LW{2,1}';
% fig 1b
figure
hold on
plot(X2(Y==0,1),X2(Y==0,2),'ro');
plot(X2(Y==1,1),X2(Y==1,2),'bo');
hold off
title('GCN Embedding for RDPG Graph','FontSize',fs)

% fig 1c and 1d: storage in Matlab.mat
ln=10;tt=30;lnn=50;
tt=1;lnn=1000;
error1=zeros(ln,1);
t0=zeros(ln,1);
t1=zeros(ln,1);
t2=zeros(ln,1);
error2=zeros(ln,1);
error3=zeros(ln,1);
berr=0.145;
for j=1:ln
n=lnn*j;
% Z=unifrnd(0,1,[n,p]);
for r=1:tt
[Adj,Y]=generateSims(2,n,2);
tic
error1(j)=error1(j)+GraphNNEvaluate(Adj,Y,0,1)/tt;
t2(j)=t2(j)+toc;
tic
error2(j)=error2(j)+GraphNNEvaluate(Adj,Y,0,2)/tt;
t0(j)=t0(j)+toc;
tic
[U,S,V]=svd(Adj);d=2;
X0=U(:,1:d)*S(1:d,1:d)^0.5;
error3(j)=error3(j)+GraphEvaluate(X0,Y,0,2)/tt;
t1(j)=t1(j)+toc;
end
end
plot(1:ln,error2,'r-',1:ln,error1,'b-',1:ln,error3,'g-',1:ln,berr*ones(1,ln),'k--','LineWidth',2)
xticks([1,5,10])
xticklabels({'50','250','500'})
ylim([0.1,0.4]);
xlabel('Sample Size','FontSize',fs)
ylabel('Classification Error','FontSize',fs)
legend('GCN','GNN','ASE * LDA','Bayes Optimal','FontSize',fs)
title('10-Fold Cross Validation','FontSize',fs)

plot(1:ln,t0,'r-',1:ln,t2,'b-',1:ln,t1,'g-','LineWidth',2)
xticks([1,5,10])
xticklabels({'1000','5000','10000'})
xlabel('Sample Size','FontSize',fs)
ylabel('Running Time','FontSize',fs)
legend('GCN','GNN','ASE * LDA','FontSize',fs)
title('10-Fold Cross Validation','FontSize',fs)


% fig 3a
p=10000;
Z=unifrnd(0,1,[n,p]);
%noise
% [mdl4,filter]=GraphNN(Adj,Y,Z,2);
% X4=[Adj*filter,Z]*mdl4.IW{1,1}'*mdl4.LW{2,1}';
% figure
% hold on
% plot(X4(Y==0,1),X4(Y==0,2),'ro');
% plot(X4(Y==1,1),X4(Y==1,2),'bo');
% title('AGCN Embedding for Signal Graph + Noise Attributes','FontSize',fs)
% hold off

[ind,corr,pval] = DCorScreening(Z,Y);
[mdl4,filter]=GraphNN(Adj,Y,Z(:,ind),2);
X4=[Adj*filter,Z(:,ind)]*mdl4.IW{1,1}'*mdl4.LW{2,1}';
figure
hold on
plot(X4(Y==0,1),X4(Y==0,2),'ro');
plot(X4(Y==1,1),X4(Y==1,2),'bo');
title('AGCN Embedding for Signal Graph + Noise Attributes','FontSize',fs)
hold off
% 
% fig 3b
t1=zeros(10,1);
t2=zeros(10,1);
t3=zeros(10,1);
t4=zeros(10,1);tt=10;
for i=1:10
    n=50*i;
    for j=1:tt
    [Adj,Y]=generateSims(2,n,2);
    p=10000;
    Z=unifrnd(0,1,[n,p]);
    %noise
    [mdl4,filter]=GraphNN(Adj,Y,Z,2);
    tmp1=mdl4.IW{1,1}'*mdl4.LW{2,1}';
    tmp2=Z*tmp1(3:end,:);
    tmp1=Adj*filter*tmp1(1:2,:);
    t1(i)=t1(i)+norm(tmp1)/tt;
    t2(i)=t2(i)+norm(tmp2)/tt;
    
    [ind,corr,pval] = DCorScreening(Z,Y);
    [mdl4,filter]=GraphNN(Adj,Y,Z(:,ind),2);
    tmp1=mdl4.IW{1,1}'*mdl4.LW{2,1}';
    if sum(ind)>0
        tmp2=Z(:,ind)*tmp1(3:end,:);
    else
        tmp2=0;
    end
    tmp1=Adj*filter*tmp1(1:2,:);
    t3(i)=t3(i)+norm(tmp1)/tt;
    t4(i)=t4(i)+norm(tmp2)/tt;
    end
end
plot(1:10,t4./t3,'r-',1:10,t2./t1,'b-','LineWidth',2)
xticks([1,6,10])
xticklabels({'50','300','500'})
legend('AGCN','AGCN no Screening','FontSize',fs)
xlabel('Sample Size','FontSize',fs)
title('Noise Divided by Signal in AGCN Embedding','FontSize',fs)

% fig 3c
ln=11;
error1=zeros(ln,1);error2=zeros(ln,1);error3=zeros(ln,1);error4=zeros(ln,1);
std1=zeros(ln,1);std2=zeros(ln,1);std3=zeros(ln,1);std4=zeros(ln,1);
n=300;tt=50;beer=0.145;
for j=1:ln
    p=30*(j-1);
    for i=1:tt
        Z=unifrnd(0,1,[n,p]);
        [Adj,Y]=generateSims(2,n,2);
        error1(j)=error1(j)+GraphNNEvaluate(Adj,Y,Z,2)/tt;
        [ind,corr,pval] = DCorScreening(Z,Y);
        error2(j)=error2(j)+GraphNNEvaluate(Adj,Y,Z(:,ind),2)/tt;
    end
end
plot(1:ln,error2,'r-',1:ln,error1,'b-',1:ln,berr*ones(1,ln),'k--','LineWidth',2)
xticks([1,6,11])
xticklabels({'0','150','300'})
ylim([0,0.7])
legend('AGCN','AGCN no Screening','Bayes Optimal','FontSize',fs)
xlabel('Noise Dimension','FontSize',fs)
ylabel('Classification Error','FontSize',fs)
title('10-Fold Cross Validation','FontSize',fs)

% fig 3d
ln=11;
error1=zeros(ln,1);error2=zeros(ln,1);
n=300;tt=30;berr=0.145;
for j=1:ln
    %     p=1*j;
    for i=1:tt
        [Adj,Y,~,Z]=generateSims(2,n,j);
        error1(j)=error1(j)+GraphNNEvaluate(Adj,Y,Z,2)/tt;
        error2(j)=error2(j)+GraphNNEvaluate(Adj,Y,0,2)/tt;
    end
end
plot(1:ln,error1,'r-',1:ln,error2,'b-',1:ln,berr*ones(1,ln),'k--','LineWidth',2)
xticks([1,5,10])
xticklabels({'1','5','10'})
ylim([0.1,0.2])
legend('AGCN','GCN (no attributes)','Bayes Optimal','FontSize',fs)
xlabel('Additional Signal Dimension','FontSize',fs)
ylabel('Classification Error','FontSize',fs)
title('10-Fold Cross Validation','FontSize',fs)

% fig 4
ln=10;tt=30;
error1=zeros(ln,1);
error2=zeros(ln,1);
% error3=zeros(ln,1);
% error4=zeros(ln,1);
for j=1:ln
    n=30*j;
    % Z=unifrnd(0,1,[n,p]);
    for r=1:tt
        [Adj,Y]=generateSims(4,n,2);
%         error1(j)=error1(j)+GraphNNEvaluate(Adj,Y,0,1)/tt;
        error1(j)=error1(j)+GraphNNEvaluate(Adj,Y,0,2)/tt;
        [Adj,Y]=generateSims(5,n,2);
%         error3(j)=error3(j)+GraphNNEvaluate(Adj,Y,0,1)/tt;
        error2(j)=error2(j)+GraphNNEvaluate(Adj,Y,0,2)/tt;
    end
end
beer=0.145;
figure
plot(1:ln,error1,'r-',1:ln,error2,'b-',1:ln,beer*ones(1,ln),'k--','LineWidth',2)
xticks([1,5,10])
xticklabels({'30','150','300'})
ylim([0.1,0.4])
legend('GCN on Distance','GCN on Kernel','Bayes Optimal','FontSize',fs)
xlabel('Sample Size','FontSize',fs)
ylabel('Classification Error','FontSize',fs)
title('10 Fold Cross Validation','FontSize',fs)


% fig 5 Wiki Data
load('Wikipedia.mat')
[error0]=GraphNNEvaluate([GEAdj,GFAdj],Label);
[error1]=GraphNNEvaluate([GEAdj,GFAdj],Label,0,1,10);
[U,S,V]=svd(GEAdj);
for d=1:10
X0=U(:,1:d)*S(1:d,1:d)^0.5;
[error2(d)]=GraphEvaluate(X0,Label,0,2);
end
[U,S,V]=svd(GFAdj);
for d=1:10
X0=U(:,1:d)*S(1:d,1:d)^0.5;
[error3(d)]=GraphEvaluate(X0,Label,0,2);
end
% error1=zeros(ln,1);
% error2=zeros(ln,1);

load('graphCElegans.mat')
[error4]=GraphNNEvaluate([Ac,Ag],vcols);
[error5]=GraphNNEvaluate([Ac,Ag],vcols,0,1,10);
[U,S,V]=svd(Ac);
for d=1:20
X0=U(:,1:d)*S(1:d,1:d)^0.5;
[error6(d)]=GraphEvaluate(X0,vcols,0,2);
end
[U,S,V]=svd(Ag);
for d=1:20
X0=U(:,1:d)*S(1:d,1:d)^0.5;
[error7(d)]=GraphEvaluate(X0,vcols,0,2);
end
% fig 4 Multiplicative Model

for i=1:4
    %     [~,Y,~,X]=generateSims(i,1000);
    %     hold on
    %     ind=(Y==1);
    %     plot(X(ind,1),X(ind,2),'r.');
    %     ind=(Y==2);
    %     plot(X(ind,1),X(ind,2),'g.');
    %     ind=(Y==3);
    %     plot(X(ind,1),X(ind,2),'k.');
    %     hold off
    numRange=100:100:1000;
    error1=zeros(10,1);error2=zeros(10,1);error3=zeros(10,1);error4=zeros(10,1);
    std1=zeros(10,1);std2=zeros(10,1);std3=zeros(10,1);std4=zeros(10,1);
    for j=1:10
        n=numRange(j);
        [error1(j),error2(j),error3(j),error4(j),std1(j),std2(j),std3(j),std4(j)]=GraphNNSim(i,10,n);
    end
    
    fpath = mfilename('fullpath');
    fpath=strrep(fpath,'\','/');
    findex=strfind(fpath,'/');
    rootDir=fpath;
    pre1=strcat(rootDir);% The folder to save figures
    filename=strcat(pre1,'EncoderSim',num2str(i));
    save(filename,'error1','error2', 'error3', 'error4','std1','std2','std3','std4','numRange');
end


[X,Y,~,~]=generateSims(4,1000);
% hold on
% ind=(Y==1);
% plot(X(ind,1),X(ind,2),'r.');
% ind=(Y==2);
% plot(X(ind,1),X(ind,2),'g.');
% hold off
[X,~]=GraphFilter(X,Y);
hold on
ind=(Y==1);
plot(X(ind,1),X(ind,2),'r.');
ind=(Y==2);
plot(X(ind,1),X(ind,2),'g.');
hold off