function w = model_2layer_wrap_param(W0, W1)
% Wrap all parameters to a vector.

w = [W0(:); W1(:)];
