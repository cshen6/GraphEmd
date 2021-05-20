function [dLdW, dLdX] = backward_graph_conv(A, X, W, dLdZ)
% Z = AXW

A_dLdZ = A' * dLdZ;
dLdW = X' * A_dLdZ;

if nargout == 1
  return;
end

dLdX = A_dLdZ * W';
