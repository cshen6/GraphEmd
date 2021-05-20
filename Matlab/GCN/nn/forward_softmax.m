function P = forward_softmax(X, which_dim)
% P = softmax(Y, which_dim)

expX = exp(X - max(X,[],which_dim)); % Use shifts to ensure numerical stability
P = expX ./ sum(expX, which_dim);
P = P + eps; % Avoid explicit zero

