function L = forward_cross_entropy(P, Y)
% L = 1/n * sum( -Y .* log(P) )

n = size(Y,1);
Y_logP = Y .* log(P);
L = - sum(Y_logP(:)) / n;
