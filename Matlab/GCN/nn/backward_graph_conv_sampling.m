function [dLdW, dLdX] = backward_graph_conv_sampling(A, X, W, p, dLdZ)
% Z = A * diag(1/p) * XW / n
%
% p is the sampling probabilities corresponding to the columns of A
% (as well as the rows of X). The elements of p do not necessarily sum
% to unity.

p = p(:);
n = length(p);
A_dLdZ_p = ((A' * dLdZ) ./ p) / n;
dLdW = X' * A_dLdZ_p;

if nargout == 1
  return;
end

dLdX = A_dLdZ_p * W';
