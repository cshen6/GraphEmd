function [w, param] = sgd_one_step(w, g, param)
% One step of SGD

alpha = param.alpha;
k = param.k;

% Vanilla SGD
%
% w = w - alpha * g;

% Step size \prop 1/sqrt(k)
%
w = w - alpha/sqrt(k) * g;
k = k + 1;

param.alpha = alpha;
param.k = k;
