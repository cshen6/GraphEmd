function dLdX = backward_sigmoid(Y, dLdY)
% Y = sigmoid(X)

dLdX = dLdY .* Y .* (1 - Y);
