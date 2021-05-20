function [Y, mask] = forward_dropout(X, p)
% Y = X .* mask

mask = ( rand(size(X)) <= p ) / p;
Y = X .* mask;
