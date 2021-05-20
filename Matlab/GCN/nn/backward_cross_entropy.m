function dLdP = backward_cross_entropy(P, Y)
% L = 1/n * sum( -Y .* log(P) )

n = size(Y,1);
dLdP = - Y ./ P / n;
