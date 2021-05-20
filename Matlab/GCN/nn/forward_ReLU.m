function Y = forward_ReLU(X)
% Y = ReLU(X)

Y = X .* (X >= 0);
