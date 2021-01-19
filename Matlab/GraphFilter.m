function [Z,filter]=GraphFilter(X,Y)

n=length(Y);
k=length(unique(Y));
if min(Y)<1
    Y=Y+1-min(Y);
end
% one-hot encoding of the class label used as the fixed filter weight
filter=zeros(n,k);
for i=1:n
    filter(i,Y(i))=1;
end
m=size(X,2);
Z=zeros(n,m);
for i=1:m/n
    Z(:,(i-1)*k+1:i*k)=X(:,(i-1)*n+1:i*n)*filter;
end