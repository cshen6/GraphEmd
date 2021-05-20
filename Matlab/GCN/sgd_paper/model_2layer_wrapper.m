function [f, P, g] = model_2layer_wrapper(A, X, AX, Y, idx0, idx1, idx2, p0, p1, w, szW0, szW1, l2_reg)
% Wrapper of model_2layer:
% - w is unwrapped to W0, W1
% - L is renamed as f
% - dLdW0, dLdW1 are wrapped to vector g
%
% szW0: Size of W0
% szW1: Size of W1

[W0, W1] = model_2layer_unwrap_param(w, szW0, szW1);

if nargout <= 2

  [f, P] = model_2layer(A, X, AX, Y, idx0, idx1, idx2, p0, p1, W0, W1, l2_reg);
  
else
  
  [f, P, dLdW0, dLdW1] = model_2layer(A, X, AX, Y, idx0, idx1, idx2, ...
                                      p0, p1, W0, W1, l2_reg);
  g = model_2layer_wrap_param(dLdW0, dLdW1);
  
end
