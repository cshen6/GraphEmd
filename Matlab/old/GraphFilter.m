function [Z,filter]=GraphEncoderEmbed(X,Y)

n=length(Y);
k=length(unique(Y));
Y=Y-min(Y)+1;
nk=zeros(k,1);
for i=1:k
    nk(i)=sum(Y==i)-1;
end
% one-hot encoding of the class label used as the fixed filter weight
filter=zeros(n,k);
for i=1:n
    filter(i,Y(i))=1;
end
for i=1:k
    filter(:,i)=filter(:,i)/nk(i);
end
Z=X*filter;

% m=size(X,2);
% Z=zeros(n,m);
% for i=1:m/n
%     Z(:,(i-1)*k+1:i*k)=X(:,(i-1)*n+1:i*n)*filter;
% end