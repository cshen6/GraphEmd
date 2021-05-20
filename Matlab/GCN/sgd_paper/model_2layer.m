function [L, P, dLdW0, dLdW1] = model_2layer(A, X, AX, Y, idx0, idx1, idx2, p0, p1, W0, W1, l2_reg)
% This code implements the 2-layer FastGCN model, which includes as a
% special case the batched GCN model when sampling is not applied. The
% precomputed matrix product AX may improve computational efficiency
% (when sampling is not applied on the input layer).
%
% A:  Normalized graph adjacency matrix.
% X:  Input node feature matrix.
% AX: Precomputed product A*X.
% Y:  One-hot node label matrix (must match size of A).
%
% idx0, idx1, idx2: Sampling indices for the input (zeroth), first,
%                   and second (output) layers. Empty means no
%                   sampling.
%
% p0, p1: Sampling probability vectors for the input (zeroth) and
%                   first layers. The sampling at the second (output)
%                   layer is always uniform as required by the
%                   cross-entropy loss. If no sampling is applied for
%                   a particular layer, the corresponding probability
%                   vector is not referenced. The length must match
%                   that of idx0 and idx1, respectively.
%
% W0: First layer parameter.
% W1: Second layer parameter.
% l2_reg: L2 regularization weight.
%
% L: Loss.
% P: Prediction (probability matrix).
% dLdW0: Gradient w.r.t. W0.
% dLdW1: Gradient w.r.t. W1.

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

Z1 = forward_ReLU(U1);                  % Z1 = ReLU(U1)

if isempty(idx1)                        % U2 = A2 * Z1 * W1
  A2 = A(:, idx2)'; % A2 = A(idx2, :);  % where A2 = A(idx2, idx1)
  U2 = forward_graph_conv(A2, Z1, W1);
else
  A2 = A(idx2, idx1);
  U2 = forward_graph_conv_sampling(A2, Z1, W1, p1);
end

Z2 = forward_softmax(U2, 2);            % Z2 = softmax(U2)

Y2 = Y(idx2, :);                        % L = cross_entropy(Z2, Y2)
L  = forward_cross_entropy(Z2, Y2);

% Forward, regularization
L = L + l2_reg * ( norm(W0, 'fro')^2 + norm(W1, 'fro')^2 );

if nargout == 1
  return;
end

P = Z2;

if nargout == 2
  return;
end

% Backward
dLdZ2 = backward_cross_entropy(Z2, Y2);  % dL = dcross_entropy(Z2, Y2) * dZ2

dLdU2 = backward_softmax(Z2, 2, dLdZ2);  % dZ2 = dsoftmax(U2) * dU2

if isempty(idx1)                                          % dU2 = A2 * Z1 * dW1
  [dLdW1, dLdZ1] = backward_graph_conv(A2, Z1, W1, dLdU2);%     + A2 * dZ1 * W1
else
  [dLdW1, dLdZ1] = backward_graph_conv_sampling(A2, Z1, W1, p1, dLdU2);
end

dLdU1 = backward_ReLU(U1, dLdZ1);        % dZ1 = dReLU(U1) * dU1

if isempty(idx0) & isempty(idx1)         % dU1 = A1 * X0 * dW0
  dLdW0 = backward_fc(AX, W0, dLdU1);
elseif isempty(idx0) & ~isempty(idx1)
  dLdW0 = backward_fc(AX1, W0, dLdU1);
else % if ~isempty(idx0)
  dLdW0 = backward_graph_conv_sampling(A1, X0, W0, p0, dLdU1);
end

% Backward, regularization
dLdW0 = dLdW0 + 2*l2_reg * W0;
dLdW1 = dLdW1 + 2*l2_reg * W1;
