function [f, P, g] = model_1layer_wrapper(A, X, AX, Y, idx0, idx1, p0, w, szW0, l2_reg)
% Wrapper of model_1layer:
% - w is unwrapped to W0
% - L is renamed as f
% - dLdW0 is wrapped to vector g
%
% szW0: Size of W0

[W0] = model_1layer_unwrap_param(w, szW0);

if nargout <= 2

  [f, P] = model_1layer(A, X, AX, Y, idx0, idx1, p0, W0, l2_reg);
  
else
  
  [f, P, dLdW0] = model_1layer(A, X, AX, Y, idx0, idx1, p0, W0, l2_reg);
  g = model_1layer_wrap_param(dLdW0);
  
end
