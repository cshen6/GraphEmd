% Test 2-layer architecture

clear all;
addpath(genpath('../nn'));

%------------------------------------------------------------------------------
% dataset: cora
%{
% Load in data
data_path = '../data';
data_name = 'cora';
load([data_path '/' data_name]);

% Hyperparameters
num_epoch = 100;   % Number of epochs
d2 = 16;           % Number of hidden units
l2_reg = 0;        % L2 regularization weight
batch_size = 256;  % Batch size

% Experiment setting: additional hyperparameters
setting = {};
legend_txt = {};
id = 1;
setting{id}.learning_rate = 1e+2;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 400;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+2;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 800;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+2;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = ceil(size(A,1)/2); % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl n/2)'];
id = id + 1;
setting{id}.learning_rate = 1e+2;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['SGD unbiased'];
id = id + 1;
setting{id}.learning_rate = 1e-1;    % Learning rate
setting{id}.learning_algo = 'adam';  % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['ADAM unbiased'];
id = id + 1;
%}
%------------------------------------------------------------------------------
% dataset: pubmed
%{
% Load in data
data_path = '../data';
data_name = 'pubmed';
load([data_path '/' data_name]);

% Hyperparameters
num_epoch = 100;   % Number of epochs
d2 = 16;           % Number of hidden units
l2_reg = 0;        % L2 regularization weight
batch_size = 256;  % Batch size

% Experiment setting: additional hyperparameters
setting = {};
legend_txt = {};
id = 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 400;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 800;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 1600;      % Sample size
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 3200;      % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = ceil(size(A,1)/2); % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl n/2)'];
id = id + 1;
setting{id}.learning_rate = 1e+1;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['SGD unbiased'];
id = id + 1;
setting{id}.learning_rate = 1e-1;    % Learning rate
setting{id}.learning_algo = 'adam';  % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['ADAM unbiased'];
id = id + 1;
%}
%------------------------------------------------------------------------------
% dataset: mixture
%%{
% Load in data
data_path = '../data';
data_name = 'mixture';
load([data_path '/' data_name]);

% Hyperparameters
num_epoch = 100;   % Number of epochs
d2 = 16;           % Number of hidden units
l2_reg = 0;        % L2 regularization weight
batch_size = 256;  % Batch size

% Experiment setting: additional hyperparameters
setting = {};
legend_txt = {};
id = 1;
setting{id}.learning_rate = 1e+0;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 400;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+0;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 800;       % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+0;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = 1600;      % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl ' num2str(setting{id}.sample_size) ')'];
id = id + 1;
setting{id}.learning_rate = 1e+0;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = ceil(size(A,1)/2); % Sample size
legend_txt{id} = [''];
legend_txt{id} =['SGD consistent (sampl n/2)'];
id = id + 1;
setting{id}.learning_rate = 1e+0;    % Learning rate
setting{id}.learning_algo = 'sgd';   % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['SGD unbiased'];
id = id + 1;
setting{id}.learning_rate = 1e-2;    % Learning rate
setting{id}.learning_algo = 'adam';  % Learning algorithm
setting{id}.sample_size = [];        % Sample size
legend_txt{id} = ['ADAM unbiased'];
id = id + 1;
%%}
%------------------------------------------------------------------------------

% Already have A as a sparse matrix. Only need normalization.
A = normalizeSparseA(A);

if strcmp(data_name, 'cora') || strcmp(data_name, 'pubmed') %------------------
  
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

  % Normalize X
  X = normalizeX(X);

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

elseif strcmp(data_name, 'mixture') %------------------------------------------
  
  % Extract dimensions
  n = size(A,1);       % Number of graph nodes
  d = size(X,2);       % Feature dimension
  c = size(Y,2);       % Number of classes
  szW0 = [d,d2];       % Size of parameter matrix W0
  szW1 = [d2,c];       % Size of parameter matrix W1

  % Already have X (dense)
  
  % No need to normalize X

  % Already have Y (dense)

  % Already have indices, but need to +1 due to python->matlab conversion
  idx_train = idx_train + 1;
  idx_valid = idx_valid + 1;
  idx_test = idx_test + 1;

  % Use length() to get ntrain, nvalid, and ntest.

end %--------------------------------------------------------------------------

% For each experiment setting
num_setting = length(setting);
loss_hist = cell(num_setting, 1);
true_loss_hist = cell(num_setting, 1);
for i = 1:num_setting
  
  % Fix seed
  RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
  
  % Params
  learning_rate = setting{i}.learning_rate;
  learning_algo = setting{i}.learning_algo;
  sample_size = setting{i}.sample_size;
  
  % Initialize optimizer
  if strcmp(learning_algo, 'sgd')
    optimizer_param = sgd_init(learning_rate);
  elseif strcmp(learning_algo, 'adam')
    num_var = prod(szW0) + prod(szW1);
    optimizer_param = adam_init(num_var, learning_rate);
  end

  % Print info
  fprintf('Data set: %s\n', data_name);
  fprintf('split: %d/%d/%d\n', ...
          length(idx_train), length(idx_valid), length(idx_test));
  fprintf('num_epoch %d\n', num_epoch);
  fprintf('hidden_size %d\n', d2);
  fprintf('learning_rate %g\n', learning_rate);
  fprintf('learning_algo %s\n', learning_algo);
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
  [loss_hist{i}, ~, true_loss_hist{i}] = ...
      model_2layer_train_and_test(...
      A, X, Y, idx_train, idx_valid, idx_test, szW0, szW1, l2_reg, ...
      num_epoch, batch_size, sample_size, learning_algo, optimizer_param);

end

% Plot results
figure(1);
fs = 20;
lw = 2;
for i = 1:length(setting)
  plot(true_loss_hist{i}, 'linewidth', lw);
  if i == 1
    hold on;
  elseif i == length(setting)
    hold off;
  end
end
set(gca, 'fontsize', fs);
legend(legend_txt);
xlabel('epoch ( k / num\_batch )');
ylabel('loss');
