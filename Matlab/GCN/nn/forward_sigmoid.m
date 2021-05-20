function Y = forward_sigmoid(X)
% Y = sigmoid(X)

Y = 1 ./ (1 + exp(-X));
