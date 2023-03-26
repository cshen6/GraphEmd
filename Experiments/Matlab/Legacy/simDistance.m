function simDistance

%%%%%%%%%%%%%%%%%%%%%%%% Distance Label fusion
D = rand(100);

% Generate random labels for 10 data points
n_labeled = 10;
labels = [ones(n_labeled, 1); zeros(size(D, 1) - n_labeled, 1)];
labels = labels(randperm(length(labels)));

% Fit a semi-supervised graph for label propagation
alpha = 0.5; % Weighting parameter between smoothness and consistency terms
W = fitsemigraph(D, labels, alpha);



%%% Distance and Kernel
load('Wiki_Data.mat')
Label=Label+1;
% [Z,W]=GraphEncoder(TE,Label,knum); %0 
% [Z,W]=GraphEncoder(TF,Label,knum); %0
% [Z,W]=GraphEncoder(GEAdj,Label,knum); %1138
% [Z,W]=GraphEncoder(GFAdj,Label,knum); %1004
opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',30); % default parameters
indices = crossvalind('Kfold',Label,10);
opts.indices=indices;opts.Learner=2;opts.LearnIter=0; %opts2.indices=indices;opts2.Learner=1;opts2.LearnIter=20;
WikiTE=GraphEncoderEvaluate(TE,Label,opts);
WikiTF=GraphEncoderEvaluate(TF,Label,opts);


opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',0,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',5,'epoch',100,'training',0.05,'activation','poslin'); % default parameters
% opts = struct('ASE',0,'LDA',0,'AEE',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'pivot',0,'deg',0,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','tansig'); % default parameters
%%% AEE and AEN:
load('graphCElegans.mat')
% knum=1; % all sim models have zero significant node. 
% [Z,W]=GraphEncoder(Ac,vcols,knum); %3 significant nodes
% GraphEncoder(Ag,vcols,knum); %2
indices = crossvalind('Kfold',vcols,5);
opts.indices=indices; opts.Learner=2;
CEAc=GraphEncoderEvaluate(Ac,vcols,opts);
CEAg=GraphEncoderEvaluate(Ag,vcols,opts);
opts.Spectral=0;opts.LearnIter=0;
CE=GraphEncoderEvaluate({Ac,Ag},vcols,opts);

load('CoraAdj.mat') %AEL / GFN K=2
GraphClusteringEvaluate(AdjOri,YOri)
load('email.mat') %k=42
GraphClusteringEvaluate(AdjOri,YOri)
load('lastfm.mat') %AEK K=17
GraphClusteringEvaluate(AdjOri,YOri)
load('polblogs.mat') %k=2
GraphClusteringEvaluate(AdjOri,YOri)