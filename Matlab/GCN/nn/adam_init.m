function param = adam_init(num_var, alpha, beta1, beta2, epsilon)
% Initialization of ADAM optimizer.

if exist('alpha','var'); param.alpha = alpha; else param.alpha = 1e-4; end
if exist('beta1','var'); param.beta1 = beta1; else param.beta1 = 0.9; end
if exist('beta2','var'); param.beta2 = beta2; else param.beta2 = 0.999; end
if exist('epsilon','var'); param.epsilon = epsilon; else param.epsilon = 1e-8; end

param.t = 0;
param.m = zeros(num_var,1);
param.v = zeros(num_var,1);

