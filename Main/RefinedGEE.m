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

function [Z,output1]=RefinedGEE(G,Y,opts)
warning ('off','all');
if nargin<3
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',false);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
% if ~isfield(opts,'DiagAugment'); opts.DiagAugment=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Discriminant'); opts.Discriminant=true; end
if ~isfield(opts,'Softmax'); opts.Softmax=false; end
if (opts.Refine || opts.Discriminant) && opts.Principal
    opts.Principal=0;
end
opts.BenchY=Y;

% opts.Refine=5;
% Pre-Processing
% [G,n]=ProcessGraph(G,opts); % process input graph
% if length(Y)~=n
%     disp('Input Sample Size does not match Input Label Size');
%     return;
% end
eps=0;

% Initial Graph Encoder Embedding
[Z,output1]=GraphEncoder(G,Y,opts);
% Refined Graph Encoder Embedding
if opts.Refine>0
    ZNew=cell(opts.Refine+1,1);
    ZNew{1}=Z;
    K=size(Z,2);Y2=Y;
    for r=1:opts.Refine
        Y2=Y2+output1.idx*K;
        [Z2,output2]=GraphEncoder(G,Y2,opts);
        if sum(output1.idx)-sum(output2.idx)<0
            r=r-1;
            break;
        else
            % Z=Z2;
            % Z=[Z,Z2];
            ZNew{r+1}=Z2;output1=output2;
            %dimClass=dimClass2;idx=idx2;%normZ=[normZ,norm2];
        end
    end
    % Z=ZNew{r+1};
    Z=horzcat(ZNew{1:r+1});
end
%     Z{i}=Z;%YNew{i}=tmpY;
%Y=YNew;
% output=struct('dimClass',dimClass,'comChoice',comChoice{1},'comScore',comChoice{2},'Y',Y,'norm',normZ,'idx',idx);
