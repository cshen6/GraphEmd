function w = model_fastgcn_wrap_param(W0, W1)
% Wrap all parameters to a vector.

w = [W0(:); W1(:)];
