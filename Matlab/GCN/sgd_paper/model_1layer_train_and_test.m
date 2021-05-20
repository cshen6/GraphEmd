function [loss_hist, grad_hist, true_loss_hist, true_grad_hist, test_accuracy] = model_1layer_train_and_test(A, X, Y, idx_train, idx_valid, idx_test, szW0, l2_reg, num_epoch, batch_size, sample_size, learning_algo, optimizer_param)
% All-in-one training and testing for model_1layer.
%
% A: Normalized graph adjacency matrix
% X: Normalized node feature matrix
% Y: One-hot node label matrix
% idx_train:       Node index for training
% idx_valid:       Node index for validation
% idx_test:        Node index for testing
% szW0:            Size of W0
% l2_reg:          L2 regularization weight
% num_epoch:       Number of epochs
% batch_size:      Batch size
% sample_size:     FastGCN sample size. If empty, equivalent to batched GCN.
% learning_algo:   Learning algorithm 
% optimizer_param: The parameters and internal values of the optimizer
%
% loss_hist:      History of loss
% grad_hist:      History of gradient norm
% true_loss_hist: History of true loss
% true_grad_hist: History of true gradient norm
% test_accuracy:  Test accuracy at best epoch

% Labels
[~, Ytrain_label] = max(Y(idx_train,:),[],2);
[~, Yvalid_label] = max(Y(idx_valid,:),[],2);
[~, Ytest_label] = max(Y(idx_test,:),[],2);

% Precompute AX
AX = A*X;

% Sampling probability
%-- L1
%{
p = full(sum(A,1));
p = p / sum(p);
%}
%-- L2
%{
p = full(sum(A.^2,1));
p = p / sum(p);
%}
%-- uniform
%%{
n = size(A,1);
p = ones(n,1) / n;
%%}

% Initialize w
W0 = initW(szW0);
w = model_1layer_wrap_param(W0);

% Storing results
loss_hist = zeros(num_epoch, 1);
grad_hist = zeros(num_epoch, 1);
true_loss_hist = zeros(num_epoch, 1);
true_grad_hist = zeros(num_epoch, 1);
valid_accuracy_best = -inf;
epoch_best = 0;
w_best = w;

% Training
tic
for epoch = 1:num_epoch
  
  % Training
  update_w = 1;
  comp_true_loss = 1;
  [loss_hist(epoch), grad_hist(epoch), true_loss_hist(epoch), ...
   true_grad_hist(epoch), ~, w, optimizer_param] = ...
      model_1layer_train_and_test_one_epoch(...
          A, X, AX, Y, p, w, szW0, l2_reg, batch_size, ...
          sample_size, learning_algo, optimizer_param, ...
          idx_train, Ytrain_label, update_w, comp_true_loss);

  % Validation
  update_w = 0;       % No training of w
  batch_size1 = [];   % Large batch size to gain efficiency
  sample_size1 = [];  % No sampling to recover accuracy
  comp_true_loss = 0; % No need to compute true loss
  [~, ~, ~, ~, valid_accuracy] = ...
      model_1layer_train_and_test_one_epoch(...
          A, X, AX, Y, p, w, szW0, l2_reg, batch_size1, ...
          sample_size1, learning_algo, optimizer_param, ...
          idx_valid, Yvalid_label, update_w, comp_true_loss);
  
  % Log the best epoch
  if valid_accuracy_best < valid_accuracy
    valid_accuracy_best = valid_accuracy;
    epoch_best = epoch;
    w_best = w;
  end
  
end
toc

% Testing at best epoch
update_w = 0;       % No training of w
batch_size1 = [];   % Large batch size to gain efficiency
sample_size1 = [];  % No sampling to recover accuracy
comp_true_loss = 0; % No need to compute true loss
[~, ~, ~, ~, test_accuracy] = ...
    model_1layer_train_and_test_one_epoch(...
        A, X, AX, Y, p, w_best, szW0, l2_reg, batch_size1, ...
        sample_size1, learning_algo, optimizer_param, ...
        idx_test, Ytest_label, update_w, comp_true_loss);

% Print result
fprintf('best epoch: %d\n', epoch_best);
fprintf('test_accuracy at best epoch %g\n', test_accuracy);

%------------------------------------------------------------------------------

function [loss, grad, true_loss, true_grad, accuracy, w, optimizer_param] = model_1layer_train_and_test_one_epoch(A, X, AX, Y, p, w, szW0, l2_reg, batch_size, sample_size, learning_algo, optimizer_param, idx, Y_label, update_w, comp_true_loss)
% Subroutine
%
% If batch_size is empty, the whole idx is used as batch.
% loss and grad (gradient norm) are the value for the last batch.
% true_loss and true_grad are loss and gradient norm w/o sampling & batching.
% accuracy is computed over all batches.

% Used to hold softmax results
P = zeros(size(Y));

% Before any update of w, compute true loss and gradient norm at the
% current w.
if comp_true_loss
  if update_w
    [true_loss, ~, g] = model_1layer_wrapper(A, X, AX, Y, [], idx, ...
                                             [], w, szW0, l2_reg);
    true_grad = norm(g);
  else
    [true_loss] = model_1layer_wrapper(A, X, AX, Y, [], idx, ...
                                       [], w, szW0, l2_reg);
    true_grad = -1;
  end
else
  true_loss = -inf;
  true_grad = -1;
end

% Batching
if isempty(batch_size)
  num_batch = 1;
else
  nidx = length(idx);
  shuffle = randperm(nidx);
  num_batch = ceil(nidx/batch_size);
end
for i = 1:num_batch
  
  % Batch index [for first (output) layer]
  if isempty(batch_size)
    idx1 = idx;
  else
    idx1 = idx(shuffle( (i-1)*batch_size+1 : min(i*batch_size,nidx) ));
  end

  % Sampling for input (zeroth) layer
  if isempty(sample_size)
    idx0 = [];
  else
    idx0 = randsample(length(p), sample_size, true, p);
  end
  p0 = p(idx0);
  
  % Compute loss f, prediction matrix PP, and optionally gradient g
  if update_w
    [f, PP, g] = model_1layer_wrapper(A, X, AX, Y, idx0, idx1, ...
                                      p0, w, szW0, l2_reg);
  else
    [f, PP] = model_1layer_wrapper(A, X, AX, Y, idx0, idx1, ...
                                   p0, w, szW0, l2_reg);
  end
  
  % Log PP
  P(idx1,:) = PP;
  
  % Update parameters
  if update_w
    if strcmp(learning_algo, 'sgd')
      [w, optimizer_param] = sgd_one_step(w, g, optimizer_param);
    elseif strcmp(learning_algo, 'adam')
      [w, optimizer_param] = adam_one_step(w, g, optimizer_param);
    end
  end
  
end

% Output results
loss = f; % Loss of the last batch
if update_w
  grad = norm(g); % Gradient norm of the last batch
else
  grad = -1;
end
accuracy = metric_accuracy(P(idx,:), Y_label); % Accuracy over all batches
