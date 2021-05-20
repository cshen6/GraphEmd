function param = sgd_init(alpha)
% Initialization of SGD optimizer.

if exist('alpha','var'); param.alpha = alpha; else param.alpha = 1e-2; end

param.k = 1;
