function [Edge]=adj2edge(Adj,directed)

if nargin<2
    directed=0;
end
n=size(Adj,1);
tmp=1;
if directed==0
    tmp=2;
end
Edge=zeros(sum(sum(Adj~=0))/tmp,3);
s=1;
for i=1:n
    tmp=1;
    if directed==0
       tmp=i+1;
    end
    for j=tmp:n
        if Adj(i,j)~=0
            Edge(s,1)=i;
            Edge(s,2)=j;
            Edge(s,3)=Adj(i,j);
            s=s+1;
        end
    end
end