function [w, param] = adam_one_step(w, g, param)
% One step of ADAM

alpha = param.alpha;
beta1 = param.beta1;
beta2 = param.beta2;
epsilon = param.epsilon;
t = param.t;
m = param.m;
v = param.v;

% Paper version
%
% t = t + 1;
% m = beta1 * m + (1-beta1) * g;
% v = beta2 * v + (1-beta2) * (g.^2);
% m_hat = m / (1 - beta1^t);
% v_hat = v / (1 - beta2^t);
% w = w - alpha * m_hat ./ (sqrt(v_hat) + epsilon);

% Tensorflow version
%
t = t + 1;
lr = alpha * sqrt(1 - beta2^t) / (1 - beta1^t);
m = beta1 * m + (1-beta1) * g;
v = beta2 * v + (1-beta2) * (g.^2);
w = w - lr * m ./ (sqrt(v) + epsilon);

param.alpha = alpha;
param.beta1 = beta1;
param.beta2 = beta2;
param.epsilon = epsilon;
param.t = t;
param.m = m;
param.v = v;
