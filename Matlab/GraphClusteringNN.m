%% Do vertex clustering based on Encoder Embedding * K-means
%%
%% @param X is either n*n adjacency, or s*2 edge list.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected.
%%        If X is edge list, it shall be unweighted.
%% @param k is desired number of clusters
%% @param maxIter is maximum number of iteration
%%
%% @return The n*1 cluster index ind
%% @return The n*k Encoder Embedding Z
%% @return The n*k Encoder Transformation W
%%
%% @export
%% 
function [ind,Z,W]=GraphClusteringNN(X,k,opts)

%%% if necessary, use the following code to remove any zero row and column from X:
%tmp=vecnorm(X); idx=(tmp>0); X=X(idx,idx);

%%% default parameter
if nargin<2
    k=2;
end
if nargin < 3
    opts = struct('maxIter',100,'neuron',10,'epoch',100,'normalize',0); % default parameters
end
if ~isfield(opts,'maxIter'); opts.maxIter=100; end
if ~isfield(opts,'neuron'); opts.neuron=10; end
if ~isfield(opts,'epoch'); opts.epoch=100; end
if ~isfield(opts,'normalize'); opts.normalize=0; end
if opts.normalize==1
    deg=diag(sum(X));
    %X=deg^-1*X;
    X=deg^-0.5*X*deg^-0.5;
end
dimension1 = k;
dimension2 = 1;
net = selforgmap([dimension1 dimension2],3,3,'gridtop','linkdist');
net.trainParam.showWindow = false;
net.trainParam.epochs=opts.epoch;

if size(X,2)==2
%     X=X-min(X)+1;
    n=max(max(X));
else
    n=size(X,1);
end
ind=randi([1,k],[n,1]);
warning ('off','all');
reseed=0;
for r=1:opts.maxIter
    Z=GraphEncoder(X,ind);
    try
        [net,~] = train(net,Z');
        indNew = net(Z');
        indNew = vec2ind(indNew)';
%         [~,~,indNew]=unique(indNew);
    catch
        %%% when a graph is very sparse, sometimes Z can have many repeated entries from a random initialization, and kmeans will fail
        %warning('Re-initialize index due to graph being too sparse')
        reseed=1;
    end
    
    if reseed==1 %|| sum(isnan(indNew))>0
       reseed=0;
       r=r+1; 
       indNew=randi([1,k],[n,1]); %%% re-initialize 
    end
    
    if RandIndex(ind,indNew)==1 && reseed==0
        break;
    else
        ind=indNew;
    end
end
[Z,W]=GraphEncoder(X,indNew);