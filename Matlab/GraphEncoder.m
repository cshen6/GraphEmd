%% Do AEE. Running time is O(s) where s is number of edges.
%%
%% @param X is either n*n adjacency, or s*2 edge list.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected. Complexity in O(n^2).
%%        If X is edge list, it shall be unweighted. Complexity in O(s).
%% @param Y is n*1 class label vector
%%
%% @return The n*k Encoder Embedding Z
%% @return The n*k Encoder Transformation W
%%
%% @export
%% 

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

% Adjacency Matrix Version in O(n^2)
if size(X,1)==n && size(X,2)==n
   Z=X*W;
end
% Edge List Version in O(s) (thus more efficient for large sparse graph)
if size(X,2)==2
%     X=X-min(X)+1;
    Z=zeros(n,k);
    s=size(X,1);
    for i=1:s
        a=X(i,1);b=X(i,2);c=Y(a);d=Y(b);
        Z(a,d)=Z(a,d)+1;
        Z(b,c)=Z(b,c)+1;
    end
    for i=1:k
        Z(:,i)=Z(:,i)/nk(i);
    end
end