function [Z,W]=GraphEncoder(X,Y)

n=length(Y);
k=length(unique(Y));
Y=Y-min(Y)+1;
nk=zeros(k,1);
for i=1:k
    nk(i)=sum(Y==i)-1;
end
% one-hot encoding of the class label used as the fixed weight
W=zeros(n,k);
for i=1:n
    W(i,Y(i))=1;
end
for i=1:k
    W(:,i)=W(:,i)/nk(i);
end
Z=X*W;

% m=size(X,2);
% Z=zeros(n,m);
% for i=1:m/n
%     Z(:,(i-1)*k+1:i*k)=X(:,(i-1)*n+1:i*n)*filter;
% end