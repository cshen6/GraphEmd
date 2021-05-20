function [Adj,Y]=edge2adj(X,Y)

% X=table2array(X);
% Y=table2array(Y);
if (min(min(X))==0)
    X=X+1;
end
n=size(Y,1);
Adj=zeros(n,n);
for i=1:size(X,1)
    Adj(X(i,1),X(i,2))=1;
end
X=Adj+Adj';