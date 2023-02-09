%% Compute the Graph Encoder Embedding.
%% Running time is O(nK+s) where s is number of edges, n is number of vertices, and K is number of class.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "One-Hot Graph Encoder Embedding", 2022.
%%
%% @param X is either n*n adjacency, or s*3 edge list. Vertex size should be >10.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected. It will be converted to s*3 edgelist.
%%        Edgelist input can be either s*2 or s*3, and complexity in O(s).
%%
function [Z,Z_d,Y,time,ind]=GraphDynamics(X,Y)

if nargin<3
    opts = struct('Common',true,'d',1);
end
if ~isfield(opts,'Common'); opts.Common=true; end
if ~isfield(opts,'d'); opts.d=1; end

%%% Dynamic Encoder Embedding and Reshape
time=zeros(3,1);
tic
[Z,Y]=GraphEncoder(X,Y);
t=length(X);
[n,Kt]=size(Z);
K=Kt/t;
Z=reshape(Z,n,K,t);
time(1)=toc;

if t>1
    %%% Extract common vertices only
    if opts.Common==true
        vnorm=zeros(size(Z,1),t);
        for i=1:t
            vnorm(:,i)=vecnorm(Z(:,:,i),2,2);
        end
        vnorm=sum(vnorm,2);
        ind=(vnorm==t);
        Z=Z(ind,:,:); %use common indices, then re-normalize
        for j=1:t
            Z(:,:,j) = normalize(Z(:,:,j),2,'norm');
        end
        time(2)=toc;
        n=size(Z,1);
        Y=Y(ind);
    end

    %%% Dynamic Vertex Embedding
    tic
    Z_d=zeros(n,t);
    for r=1:n
%         tmpDist=zeros(t,t);
        tmpDist=zeros(t,1);
        for i=1:t
            tmpDist(i,1)=norm(Z(r,:,1)-Z(r,:,i),'fro');
        end
%         for i=1:t
%             for j=i+1:t
%                 tmpDist(i,j)=norm(Z(r,:,i)-Z(r,:,j),'fro');
%                 tmpDist(j,i)=tmpDist(i,j);
%             end
%         end
%         Z_d(r,:)=cmdscale(tmpDist,opts.d);
          [U,S,~]=svds(tmpDist,1);
          Z_d(r,:)=(U(:,1)*S^0.5)';
    end
    % Z_diff=max(Z_d,[],2)-min(Z_d,[],2);
    % ind_out=find(Z_diff>1);
    time(3)=toc;
end
