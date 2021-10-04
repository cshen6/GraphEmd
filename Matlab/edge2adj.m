function [Adj]=edge2adj(Edge,directed)

% X=table2array(X);
% Y=table2array(Y);
if nargin<2
    directed=0;
end
weighted=0;
n=max(max(Edge(:,1:2)));
if (min(min(Edge))==0)
    Edge=Edge+1; %add up the edge index
end
% n=size(Y,1);
Adj=zeros(n,n);
if length(size(Edge))==3
    weighted=1;
end
for i=1:size(Edge,1)
    tmp=1;
    if weighted==1
        tmp=Edge(i,3);
    end
    Adj(Edge(i,1),Edge(i,2))=tmp;
    if directed==0
       Adj(Edge(i,2),Edge(i,1))=tmp;
    end
end
% Adj=Adj+Adj';