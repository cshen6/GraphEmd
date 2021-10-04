function [Z2,indT,Prob,B,pi,theta]=GraphSBMEst(Adj,Y)

[Z,indT,Y]=GraphEncoder(Adj,Y); %indT is training data
K=max(Y);B=zeros(K,K);pi=zeros(K,1);n=sum(indT);
D=sum(Adj,2); %D=D/n; 
dMean=zeros(K,1);
% ns=100;
% theta=zeros(K,ns,2);
theta=cell(K,2);
Prob=zeros(n,K);
for i=1:K
    ind=(Y==i);
    Prob(ind,i)=1;
    pi(i)=sum(ind)/n;
    tmp=D(ind);
    dMean(i)=mean(tmp);
    tmp=tmp/dMean(i);
    D(ind)=tmp;
    [theta{i,1},theta{i,2}] = ecdf(tmp); %experimental cdf
%     ind=1:(length(f)-1)/(ns-1):length(f);
%     ind=ceil(ind);
%     theta{i,1}=f;theta{i,2}=x;
%     [theta(i,:,1),theta(i,:,2)]=ksdensity(tmp,'Function','cdf','NumPoints',ns);
end

% D(~indT)=D(~indT)/(dMean'*pi);
% theta=D;
% [F,XI]=ksdensity(D,'Support','positive','function','cdf');
% theta=[F;XI];

% D=D./max(D);
% dk=[mean(D(Y==1))];
% for i=2:K
%     dk=[dk,mean(D(Y==i))];
% end
Z2=Z./repmat(D,1,K);%.*repmat(dk.^2,n,1);
Z2(D==0,:)=0;
% D=sum(Adj,2); D=D./max(D);
% dk=[mean(D(Y==1)),mean(D(Y==2))];
% Z=Z./repmat(D,1,2)./repmat(dk,n,1);
% Z=Z/0.3;
% Z=Z./repmat(theta,1,2);
for i=1:K
    B(i,:)=mean(Z2(Y==i,:));
end

indTest=find(indT==0);
for j=1:length(indTest)
    tmp=zeros(K,1);
    tt=indTest(j);
    for i=1:K
        tmp(i)=norm(Z(tt,:)/D(tt)*dMean(i)-B(i,:));
    end
%      tmp
    [~,k]=min(tmp);
    Z2(tt,:)=Z(tt,:)/D(tt)*dMean(k);
end
