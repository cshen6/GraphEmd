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

function [Z,W,Z2]=GraphEncoder(X,Y)

n=length(Y);
[tmp,~,Y]=unique(Y);
k=length(tmp);
nk=zeros(k,1);
for i=1:k
    nk(i)=sum(Y==i);
end

% the class label encoding
W=zeros(n,k);
for i=1:n
    W(i,Y(i))=1;
end
for i=1:k
    W(:,i)=W(:,i)/nk(i);
end

num=size(X,3); Z2=0;
% Adjacency Matrix Version in O(n^2)
if size(X,1)==n && size(X,2)==n
    if num==1
        Z=X*W;
    else
        Z=zeros(n,k,num);
        for r=1:num
            Z(:,:,r)=X(:,:,r)*W;
        end
        Z2=Z;
        Z=reshape(Z2,n,k*num);
    end
end
% Edge List Version in O(s) (thus more efficient for large sparse graph)
if size(X,2)==2 && num==1
%     X=X-min(X)+1;
    Z=zeros(n,k);
    s=size(X,1);
    for i=1:s
        a=X(i,1);b=X(i,2);c=Y(a);d=Y(b);
        Z(a,d)=Z(a,d)+W(b,d);
        Z(b,c)=Z(b,c)+W(a,c);
    end
end

% % Check if any dimension has NaN value
% ind=(sum(isnan(Z),1)==0); 
% Z=Z(:,ind);W=W(:,ind);