%%% for undirected and unweighted graph

function [Adj]=edge2adj(Edge)

% X=table2array(X);
% Y=table2array(Y);
n=max(max(Edge));
if (min(min(Edge))==0)
    Edge=Edge+1; %add up the edge index
end
% n=size(Y,1);
Adj=zeros(n,n);
for i=1:size(Edge,1)
    Adj(Edge(i,1),Edge(i,2))=1;
end
Adj=Adj+Adj';