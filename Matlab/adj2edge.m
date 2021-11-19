
function [Edge]=adj2edge(Adj)
n=size(Adj,1);
Adj=Adj+Adj';
ss=sum(diag(Adj)>0);
Edge=zeros((sum(sum(Adj>0))-ss)/2,2);
% symc=issymmetric(Adj);
% if symc==true
%     Edge=zeros(sum(sum(Adj>0))/2,2);
% else
%     Edge=zeros(sum(sum(Adj>0)),2);
% end
s=1;
for i=1:n
%     if symc==true
%         st=i+1;
%     else
%         st=1;
%     end
    for j=i+1:n
        if Adj(i,j)>0
            Edge(s,1)=i;
            Edge(s,2)=j;
            s=s+1;
        end
    end
end
% Edge=Edge(Edge(:,1)>0,:);