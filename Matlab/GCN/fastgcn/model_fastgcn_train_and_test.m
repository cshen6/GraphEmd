function [test_accuracy_best]=model_fastgcn_train_and_test(A, X, Y, idx_train, idx_valid, idx_test, szW0, szW1, l2_reg, num_epoch, batch_size, sample_size, adam_param)
% All-in-one training and testing for model_fastgcn.
%
% A: Normalized graph adjacency matrix
% X: Normalized node feature matrix
% Y: One-hot node label matrix
% idx_train:   Node index for training
% idx_valid:   Node index for validation
% idx_test:    Node index for testing
% szW0:        Size of W0
% szW1:        Size of W1
% l2_reg:      L2 regularization weight
% num_epoch:   Number of epochs
% batch_size:  Batch size. See note below.
% sample_size: FastGCN sample size. See note below.
% adam_param:  The parameters and internal values of the ADAM optimizer
%
% Note: If both batch_size and sample_size are empty, the model is
% equivalent to GCN w/o batching. If batch_size is not empty but
% sample_size is empty, then the model is equivalent to batched GCN.

% Labels
[~, Ytrain_label] = max(Y(idx_train,:),[],2);
[~, Yvalid_label] = max(Y(idx_valid,:),[],2);
[~, Ytest_label] = max(Y(idx_test,:),[],2);

% Precompute AX
AX = A*X;

% Sampling probability
%-- L1
%%{
p = full(sum(A,1));
p = p / sum(p);
%%}
%-- L2
%{
p = full(sum(A.^2,1));
p = p / sum(p);
%}
%-- uniform
%{
n = size(A,1);
p = ones(n,1) / n;
%}

% Initialize w
W0 = initW(szW0);
W1 = initW(szW1);
w = model_fastgcn_wrap_param(W0, W1);

% Storing results
train_loss = zeros(num_epoch, 1);
valid_accuracy = zeros(num_epoch, 1);
valid_accuracy_best = -inf;
epoch_best = 0;
w_best = w;

% SGD
tic
for epoch = 1:num_epoch
  
  % Training
  update_w = 1;
  [train_loss(epoch), ~, w, adam_param] = ...
      model_fastgcn_train_and_test_one_epoch(...
          A, X, AX, Y, p, w, szW0, szW1, l2_reg, batch_size, ...
          sample_size, adam_param, idx_train, Ytrain_label, update_w);

  % Validation
  update_w = 0;      % No training of w
  batch_size1 = [];  % Large batch size to gain efficiency
  sample_size1 = []; % No sampling to recover accuracy
  [~, valid_accuracy(epoch)] = ...
      model_fastgcn_train_and_test_one_epoch(...
          A, X, AX, Y, p, w, szW0, szW1, l2_reg, batch_size1, ...
          sample_size1, adam_param, idx_valid, Yvalid_label, update_w);
  
  % Log the best epoch
  if valid_accuracy_best < valid_accuracy(epoch)
    valid_accuracy_best = valid_accuracy(epoch);
    epoch_best = epoch;
    w_best = w;
  end
  
end
toc

% Testing
update_w = 0;      % No training of w
batch_size1 = [];  % Large batch size to gain efficiency
sample_size1 = []; % No sampling to recover accuracy
[~, test_accuracy] = ...
    model_fastgcn_train_and_test_one_epoch(...
        A, X, AX, Y, p, w, szW0, szW1, l2_reg, batch_size1, ...
        sample_size1, adam_param, idx_test, Ytest_label, update_w);

% Testing at best epoch
update_w = 0;      % No training of w
batch_size1 = [];  % Large batch size to gain efficiency
sample_size1 = []; % No sampling to recover accuracy
[~, test_accuracy_best] = ...
    model_fastgcn_train_and_test_one_epoch(...
        A, X, AX, Y, p, w_best, szW0, szW1, l2_reg, batch_size1, ...
        sample_size1, adam_param, idx_test, Ytest_label, update_w);

% % % Print result
fprintf('valid_accuracy %g\n', valid_accuracy(end));
fprintf('test_accuracy %g\n', test_accuracy);
fprintf('best epoch: %d\n', epoch_best);
fprintf('valid_accuracy at best epoch %g\n', valid_accuracy(epoch_best));
fprintf('test_accuracy at best epoch %g\n', test_accuracy_best);

% % Plot history
% lw = 2;
% fs = 24;
% ms = 10;
% epoch_list = (1:num_epoch)';
% plot(epoch_list, train_loss, 'r-', 'linewidth', lw, 'markersize', ms);
% hold on;
% plot(epoch_list, valid_accuracy, 'm-', 'linewidth', lw, 'markersize', ms);
% hold off;
% legend('train batch loss', 'valid accuracy');
% set(gca, 'fontsize', fs);
% xlabel('# epochs');

%------------------------------------------------------------------------------

function [loss, accuracy, w, adam_param] = model_fastgcn_train_and_test_one_epoch(A, X, AX, Y, p, w, szW0, szW1, l2_reg, batch_size, sample_size, adam_param, idx, Y_label, update_w)
% Subroutine
%
% If batch_size is empty, the whole idx is used as batch.

% Used to hold softmax results
P = zeros(size(Y));

% Batching
if isempty(batch_size)
  num_batch = 1;
else
  nidx = length(idx);
  shuffle = randperm(nidx);
  num_batch = ceil(nidx/batch_size);
end
for i = 1:num_batch
  
  % Batch index [for second (output) layer]
  if isempty(batch_size)
    idx2 = idx;
  else
    idx2 = idx(shuffle( (i-1)*batch_size+1 : min(i*batch_size,nidx) ));
  end

  % Sampling for first layer
  if isempty(sample_size)
    idx1 = [];
  else
    idx1 = randsample(length(p), sample_size, true, p);
  end
  p1 = p(idx1);
  
  % Sampling is disabled for input (zeroth) layer
  idx0 = [];
  p0 = [];
  
  % Compute loss f, prediction matrix PP, and optionally gradient g
  if update_w
    [f, PP, g] = model_fastgcn_wrapper(A, X, AX, Y, idx0, idx1, idx2, ...
                                       p0, p1, w, szW0, szW1, l2_reg);
  else
    [f, PP] = model_fastgcn_wrapper(A, X, AX, Y, idx0, idx1, idx2, ...
                                    p0, p1, w, szW0, szW1, l2_reg);
  end
  
  % Log PP
  P(idx2,:) = PP;
  
  % Update parameters
  if update_w
    [w, adam_param] = adam_one_step(w, g, adam_param);
  end
  
end

% Output results
loss = f; % Loss of the last batch
accuracy = metric_accuracy(P(idx,:), Y_label); % Accuracy over all batches
