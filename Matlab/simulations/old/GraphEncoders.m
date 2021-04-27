%% Compute the Adjacency Encoder Embedding. 
%% Running time is O(s) where s is number of edges.
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

function [Z,W]=GraphEncoders(X,Y,rep,prob)

if nargin<3 || rep <1
    rep=10;
end
if nargin<4 || prob<=0 || prob >=1
    prob=0.8;
end

n=length(Y);
[tmp,~,Y]=unique(Y);
k=length(tmp);
nk=zeros(k,1);
for i=1:k
    nk(i)=sum(Y==i);
end
nk=nk*prob;

%% change adjacency to edge
if size(X,2)==n && size(X,1)==n
    Edge=zeros(n^2,1);s=0;
    for i=1:n
        for j=i+1:n
            if X(i,j)>0
                s=s+1;
                Edge(s,1)=i;
                Edge(s,2)=j;
            end
        end
    end
    X=Edge(1:s,:);
end

% one-hot encoding of the class label used as the fixed weight

W=zeros(n,k,rep);
for i=1:n
    W(i,Y(i),:)=(rand([1,1,rep])<prob);
end
for i=1:k
    W(:,i,:)=W(:,i,:)./nk(i);
end

Z=zeros(n,k,rep);
for i=1:s
    a=X(i,1);b=X(i,2);c=Y(a);d=Y(b);
    Z(a,d,:)=Z(a,d,:)+W(b,d,:);
    Z(b,c,:)=Z(b,c,:)+W(a,c,:);
end

% Z=reshape(Z,n,k*rep);
% W=reshape(W,n,k*rep);