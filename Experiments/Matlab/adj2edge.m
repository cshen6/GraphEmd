%% Adj to Edge Function
function [Edge,s,n]=adj2edge(Adj)
if size(Adj,2)<=3
    Edge=Adj;
    return;
end
n=size(Adj,1);
% ind=ones(n,1);
Edge=zeros(sum(sum(Adj>0)),3);
s=1;
for i=1:n
%     if sum(Adj(i,:))+sum(Adj(:,i))==0
%         ind(i)=0;
%     end
    for j=1:n
        if Adj(i,j)>0
            Edge(s,1)=i;
            Edge(s,2)=j;
            Edge(s,3)=Adj(i,j);
            s=s+1;
        end
    end
end
% Edge(s,1)=n;Edge(s,2)=n;Edge(s,3)=1;
s=s-1;