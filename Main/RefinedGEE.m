%% Compute the Graph Encoder Embedding.
%% Running time is O(nK+s) where s is number of edges, n is number of vertices, and K is number of class.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "One-Hot Graph Encoder Embedding", 2022.
%%
%% @param X is either n*n adjacency, s*2 or s*3  edge list, or a cell of edgelists that share same vertex set.
%% @param Y can be either an n*1 class label vector, or a positive integer for number of clusters, or a cell array of multiple labels and multiple cluster choice.
%%        In case of partial known labels, Y should be a n*1 vector with unknown labels set to <=0 and known labels being >0.
%%        When there is no known label, set Y to be the number of clusters or a range of clusters, Y={2,3,4};
%% @param U is an n*d node attributes
%% @param opts specifies options:
%%        Normalize specifies whether to normalize each embedding by L2 norm;
%%        Laplacian specifies whether to uses graph Laplacian or adjacency matrix;
%%        Refinement specififies whether the labels are refined by classification or clustering, 
%%                   default 0 means no refinement, 1 for refinement at current dimension, and other integers for refinement into another dimension.
%%        Directed specifices whether to output directed embedding: 0 means overall embedding, 1 means sender embedding, 2 means receiver embedding.
%%        Three integers for clustering refinement: Replicates denotes the number of replicates for clustering,
%%                                       MaxIter denotes the max iteration within each replicate for encoder embedding,
%%                                       MaxIterK denotes the max iteration used within kmeans.
%%
%% @return The n*k Encoder Embedding Z and the n*1 label vector Y. 
%% @return The n*1 boolean vector indT denoting known labels.
%% @return The GEE Clustering Score (called Minimal Rank Index in paper): ranges in [0,1] and the smaller the better (only for clustering);
%%         In case of multiple graphs, all outputs become are cell array.
%%
%% @export
%%

function [Z,output]=RefinedGEE(G,Y,opts)
warning ('off','all');
if nargin<3
    opts = struct('Normalize',true,'RefineK',3,'RefineY',3,'eps',0.2,'epsn',5);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'RefineK'); opts.RefineK=3; end
if ~isfield(opts,'RefineY'); opts.RefineY=3; end
if ~isfield(opts,'eps'); opts.eps=0.2; end
if ~isfield(opts,'epsn'); opts.epsn=5; end
opts.Discriminant = true;
opts.Principal=0;
% opts.BenchY=Y;

% opts.Refine=5;
% Pre-Processing
% [G,n]=ProcessGraph(G,opts); % process input graph
% if length(Y)~=n
%     disp('Input Sample Size does not match Input Label Size');
%     return;
% end

% Initial Graph Encoder Embedding
[Z,output]=GraphEncoder(G,Y,opts);
K=size(Z,2);

% Refined Graph Encoder Embedding
if opts.RefineK>0
    ZK=cell(opts.RefineK,1);
    output1=output;idx=output1.idx;
    for rK=1:opts.RefineK
        Y1=output1.YVal+output1.idx*K;
        [Z2,output2]=GraphEncoder(G,Y1,opts);
        % [sum(idx),sum(output2.idx & idx)]
        if sum(idx)-sum(output2.idx & idx)<= max(sum(idx)*opts.eps,opts.epsn)
            break;
        else
            ZK{rK,1}=Z2;output1=output2;idx=output2.idx & idx;
        end
    end
    ZK=horzcat(ZK{:});
    Z=[Z,ZK];
end

if opts.RefineY>0
    ZY=cell(opts.RefineY,1);
    output1=output;idx=output1.idx;
    for r=1:opts.RefineY
        [Z2,output2]=GraphEncoder(G,output1.YVal,opts);
        if sum(idx)-sum(output2.idx & idx)<= max(sum(idx)*opts.eps,opts.epsn)
            break;
        else
            ZY{r}=Z2;output1=output2;idx=output2.idx & idx;
        end
    end
    ZY=horzcat(ZY{:});
    Z=[Z,ZY];
end