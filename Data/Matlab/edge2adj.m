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
Edge=Adj+Adj';

function [Edge]=adj2edge(Adj)
n=size(Adj,1);
Edge=zeros(sum(sum(Adj))/2,2);
s=1;
for i=1:n
    for j=i+1:n
        if Adj(i,j)==1
            Edge(s,1)=i;
            Edge(s,2)=j;
            s=s+1;
        end
    end
end