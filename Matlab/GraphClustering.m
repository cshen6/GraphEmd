function [ind,Z]=GraphClustering(X,k,maxIter)

if nargin<2
    k=2;
end
if nargin<3
    maxIter=20;
end

n=size(X,1);
ind=randi([1,k],[n,1]);

for r=1:maxIter
    Z=GraphEncoder(X,ind);
    indTmp = kmeans(Z, k);
    if RandIndex(ind,indTmp)==1
        break;
    else
        ind=indTmp;
    end
end
Z=GraphEncoder(X,ind);

% m=size(X,2);
% Z=zeros(n,m);
% for i=1:m/n
%     Z(:,(i-1)*k+1:i*k)=X(:,(i-1)*n+1:i*n)*filter;
% end