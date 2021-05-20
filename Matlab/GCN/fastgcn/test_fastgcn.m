% Test FastGCN with benchmark datasets.

clear all;
addpath(genpath('../nn'));
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));

%------------------------------------------------------------------------------
% dataset: cora
%%{
% Load in data
data_path = '../data';
data_name = 'cora';
load([data_path '/' data_name]);

% Hyperparameters
num_epoch = 20;        % Number of epochs
d2 = 16;               % Number of hidden units
learning_rate = 1e-1;  % The alpha parameter in the ADAM optimizer
l2_reg = 0;            % L2 regularization weight
batch_size = 256;      % Batch size. If empty, equivalent to GCN w/o batching
%batch_size = [];
sample_size = 400;     % Sample size. If empty, equivalent to batched GCN
%sample_size = [];
%%}
%------------------------------------------------------------------------------
% dataset: pubmed
%{
% Load in data
data_path = '../data';
data_name = 'pubmed';
load([data_path '/' data_name]);

% Hyperparameters
num_epoch = 20;        % Number of epochs
d2 = 16;               % Number of hidden units
learning_rate = 1e-1;  % The alpha parameter in the ADAM optimizer
l2_reg = 0;            % L2 regularization weight
batch_size = 1024;     % Batch size. If empty, equivalent to GCN w/o batching
%batch_size = [];
sample_size = 400;     % Sample size. If empty, equivalent to batched GCN
%sample_size = [];
%}
%------------------------------------------------------------------------------

% Already have A as a sparse matrix. Only need normalization.
A = normalizeSparseA(A);

% Extract dimensions
n = size(A,1);       % Number of graph nodes
ntrain = size(y,1);  % Number of training nodes
ntest = size(ty,1);  % Number of test nodes
d = size(x,2);       % Feature dimension
c = size(y,2);       % Number of classes
szW0 = [d,d2];       % Size of parameter matrix W0
szW1 = [d2,c];       % Size of parameter matrix W1

% Assemble X
n2 = n - ntest;
X = zeros(n,d);
X(1:n2,:) = full(allx);
X(index+1,:) = full(tx); % index needs to +1

% Assemble Y
Y = zeros(n,c);
Y(1:n2,:) = full(ally);
Y(index+1,:) = full(ty); % index needs to +1

% Compute indices (GCN split)
% idx_train = 1 : ntrain;
% idx_valid = ntrain+1 : ntrain+500;
% idx_test = n-ntest+1 : n;

% Compute indices (FastGCN split)
idx_train = 1 : n-ntest-500;
idx_valid = n-ntest-500+1 : n-ntest;
idx_test = n-ntest+1 : n;

% ntrain is no longer correct under FastGCN split. Use length() to
% get the correct ntrain, nvalid, and ntest.

% Normalize X
X = normalizeX(X);

% Initialize ADAM
num_var = prod(szW0) + prod(szW1);
adam_param = adam_init(num_var, learning_rate);

% Print info
fprintf('Data set: %s\n', data_name);
fprintf('split: %d/%d/%d\n', ...
        length(idx_train), length(idx_valid), length(idx_test));
fprintf('num_epoch %d\n', num_epoch);
fprintf('hidden_size %d\n', d2);
fprintf('learning_rate %g\n', learning_rate);
fprintf('l2_regularization %g\n', l2_reg);
if isempty(batch_size)
  fprintf('batch_size []\n');
else
  fprintf('batch_size %d\n', batch_size);
end
if isempty(sample_size)
  fprintf('sample_size []\n');
else
  fprintf('sample_size %d\n', sample_size);
end

% Run
figure(2);
model_fastgcn_train_and_test(A, X, Y, idx_train, idx_valid, idx_test, ...
                             szW0, szW1, l2_reg, num_epoch, batch_size, ...
                             sample_size, adam_param);
