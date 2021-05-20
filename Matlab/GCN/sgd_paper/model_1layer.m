function [L, P, dLdW0] = model_1layer(A, X, AX, Y, idx0, idx1, p0, W0, l2_reg)
% This code implements the 1-layer FastGCN model, which includes as a
% special case the batched GCN model when sampling is not applied. The
% precomputed matrix product AX may improve computational efficiency
% (when sampling is not applied on the input layer).
%
% A:  Normalized graph adjacency matrix.
% X:  Input node feature matrix.
% AX: Precomputed product A*X.
% Y:  One-hot node label matrix (must match size of A).
%
% idx0, idx1: Sampling indices for the input (zeroth) and first
%             (output) layers. Empty means no sampling.
%
% p0: Sampling probability vector for the input (zeroth) layer. The
%             sampling at the first (output) layer is always uniform
%             as required by the cross-entropy loss. If no sampling is
%             applied, p0 is not referenced. The length must match
%             that of idx0.
%
% W0: First layer parameter.
% l2_reg: L2 regularization weight.
%
% L: Loss.
% P: Prediction (probability matrix).
% dLdW0: Gradient w.r.t. W0.

% Forward
if isempty(idx0) & isempty(idx1)        % U1 = A1 * X0 * W0
  U1 = forward_fc(AX, W0);              % where A1 = A(idx1, idx0)
elseif isempty(idx0) & ~isempty(idx1)   %       X0 = X(idx0, :)
  AX1 = AX(idx1, :);
  U1 = forward_fc(AX1, W0);
else % if ~isempty(idx0)
  if isempty(idx1)
    A1 = A(:, idx0);
  else
    A1 = A(idx1, idx0);
  end
  X0 = X(idx0, :);
  U1 = forward_graph_conv_sampling(A1, X0, W0, p0);
end

Z1 = forward_softmax(U1, 2);            % Z1 = softmax(U1)

Y1 = Y(idx1, :);                        % L = cross_entropy(Z1, Y1)
L  = forward_cross_entropy(Z1, Y1);

% Forward, regularization
L = L + l2_reg * norm(W0, 'fro')^2;

if nargout == 1
  return;
end

P = Z1;

if nargout == 2
  return;
end

% Backward
dLdZ1 = backward_cross_entropy(Z1, Y1);  % dL = dcross_entropy(Z1, Y1) * dZ1

dLdU1 = backward_softmax(Z1, 2, dLdZ1);  % dZ1 = dsoftmax(U1) * dU1

if isempty(idx0) & isempty(idx1)         % dU1 = A1 * X0 * dW0
  dLdW0 = backward_fc(AX, W0, dLdU1);
elseif isempty(idx0) & ~isempty(idx1)
  dLdW0 = backward_fc(AX1, W0, dLdU1);
else % if ~isempty(idx0)
  dLdW0 = backward_graph_conv_sampling(A1, X0, W0, p0, dLdU1);
end

% Backward, regularization
dLdW0 = dLdW0 + 2*l2_reg * W0;
