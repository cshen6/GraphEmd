function [W0, W1] = model_2layer_unwrap_param(w, szW0, szW1)
% Unwrap a vector of all parameters to individual shapes.

W0 = reshape( w(1:prod(szW0)), szW0 );
W1 = reshape( w(1+prod(szW0):end), szW1 );
