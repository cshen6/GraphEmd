function [dLdW, dLdX] = backward_fc(X, W, dLdZ)
% Z = XW

dLdW = X' * dLdZ;

if nargout == 1
  return;
end

dLdX = dLdZ * W';
