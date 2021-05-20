function Z = forward_graph_conv_sampling(A, X, W, p)
% Z = A * diag(1/p) * XW / n
%
% p is the sampling probabilities corresponding to the columns of A
% (as well as the rows of X). The elements of p do not necessarily sum
% to unity.

p = p(:);
n = length(p);
Z = A * (X ./ p) * W / n;
