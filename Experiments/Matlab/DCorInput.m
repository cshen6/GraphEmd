%% Process the input into proper distance matrix or kernel induced distance matrix, depending on choice of the metric / kernel.
%%
%% The kernel transformation is based on this paper: 
%% Cencheng Shen, Joshua T Vogelstein, "The exact equivalence of distance and kernel methods in hypothesis testing", AStA Advances in Statistical Analysis, 2021
%%
%% @param X can be n*p data, n*n dissimilarity, or n*n similarity.
%% @param optionMetric is a string that specifies which metric to use, including 'euclidean'(default),'hsic', and other variants.
%%
%% @return X as the n by n processed distance matrix.
%%
%% @export
%%

function [X]=DCorInput(X,optionMetric)
if nargin<2
    optionMetric='euclidean';  % use euclidean distance by default
end
[X,ind]=checkDist(X); % check whether it is a distance or kernel matrix
if ind>0 % if the input is already a distance or kernel matrix
    return; % return the distance or kernel induced distance matrix, 
else % form the Euclidean distance matrix and proceed further
    X=squareform(pdist(X));
end

% for the kernel choice, they are transformed to kernel induced distance
% matrix via 1-X/max(max(X)). As all kernel choice below are translation
% invariant and equals 1 in maximum, it is just 1-X
if strcmpi(optionMetric,'gaussian01')
    deg = max(max(X))*0.1;
    X=exp(-X.^2/deg^2);
    X=1-X;
end
if strcmpi(optionMetric,'laplace01')
    deg = max(max(X))*0.1;
    X=exp(-X/deg);
    X=1-X;
end
if strcmpi(optionMetric,'gaussianMax')
    deg = max(max(X));
    X=exp(-X.^2/deg^2);
    X=1-X;
end
if strcmpi(optionMetric,'laplaceMax')
    deg = max(max(X));
    X=exp(-X/deg);
    X=1-X;
end
if strcmpi(optionMetric,'laplace')
    deg = median(X(X>0));
    X=exp(-X/deg);
    X=1-X;
end
if strcmpi(optionMetric,'hsic')
    deg = sqrt(0.5*median(X(X>0)).^2);
    X=exp(-X.^2/2/deg^2);
    X=1-X;
end
if strcmpi(optionMetric,'euclidean01')
    X=X.^0.1;
end
if strcmpi(optionMetric,'euclidean19')
    X=X.^1.9;
end
if strcmpi(optionMetric,'euclidean2') || strcmpi(optionMetric,'pearson')
    X=X.^2;
end

% scale=0;
% if scale==1
%     X=(X-min(min(X)))/(max(max(X))-min(min(X)));
% end
