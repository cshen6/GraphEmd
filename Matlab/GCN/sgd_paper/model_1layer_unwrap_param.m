function [W0] = model_1layer_unwrap_param(w, szW0)
% Unwrap a vector of all parameters to individual shapes.

W0 = reshape( w(1:prod(szW0)), szW0 );
