function [Adj,Y]=GraphSBMGen(pi,B,thetaF,n)

K=length(pi);
% K=3;
% Bl=zeros(K,K);
% %             Bl=rand(clas,clas);
% Bl(:,1)=[0.9,0.1,0.1];
% Bl(:,2)=[0.1,0.5,0.1];
% Bl(:,3)=[0.1,0.1,0.2];
Adj=zeros(n,n);
tt=rand([n,1]);
Y=ones(n,1);
thres=0;

% thetaF=theta;
% theta=unifrnd(0,1,1,n);
% % quantiles = linspace(0,1,n); 
% [f,x] = ecdf(thetaF); %experimental cdf
% theta = interp1(f,x,theta,'next');%Inverse experimental cdf 
% pdnorm = makedist('Normal'); transf = icdf(pdnorm,f); %create equivalent cdf values for N(0,1) distribution


% theta=betarnd(1,4,n,1);
% theta=unifrnd(0.5,1.5,n,1);
%         theta=theta;
for i=1:K
    thres=thres+pi(i);
    Y=Y+(tt>thres); %determine the block of each data
end
theta=zeros(n,1);
for i=1:K
    ind=(Y==i);
    tmp=unifrnd(0,1,1,sum(ind));
    % quantiles = linspace(0,1,n);
%     [f,x] = ecdf(thetaF); %experimental cdf
    theta(ind) = interp1(thetaF{i,1},thetaF{i,2},tmp,'next');%Inverse experimental cdf
end
for i=1:n
    Adj(i,i)=0;%diagonals are zeros
    for j=i+1:n
        Adj(i,j)=rand(1)<theta(i)*theta(j)*B(Y(i),Y(j));
        Adj(j,i)=Adj(i,j);
    end
end