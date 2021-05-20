% Generate a mixture of c Gaussians that significantly overlap with
% each other. Points in the same class have a higher probability to be
% connected than do points in different classes.

clear all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));

% Generate points X
d = 2;

c = 3;
n0 = 2000; % Number of points per class
n = c * n0;

mu{1} = [-1/2, 0];      sigma{1} = 1.5/2;
mu{2} = [+1/2, 0];      sigma{2} = 1.0/2;
mu{3} = [0, sqrt(3)/2]; sigma{3} = 0.5/2;

for i = 1:c
  Xc{i} = randn(n0, d) .* sigma{i} + mu{i};
end

X = [];
for i = 1:c
  X = [X; Xc{i}];
end

n0_valid = 400;
n0_test = 800;
n0_train = n0 - n0_valid - n0_test;

% Y
Y = zeros(n, c);
for i = 1:c
  Y((i-1)*n0+1 : i*n0, i) = 1;
end

% Train/valid/test split. To conform with data sets processed by
% python, the indices here are 0-based. When loaded with matlab, one
% must do +1.
idx_train = [];
idx_valid = [];
idx_test = [];
for i = 1:c
  idx_train = [idx_train, (i-1)*n0 : (i-1)*n0+n0_train-1];
  idx_valid = [idx_valid, (i-1)*n0+n0_train : (i-1)*n0+n0_train+n0_valid-1];
  idx_test = [idx_test, i*n0-n0_test : i*n0-1];
end

% Generate graph
p_inter = 0.0002;
p_intra = 0.001;
R = rand(n);
A = (R < p_inter);
R0 = rand(n0);
A0 = (R0 < p_intra);
for i = 1:c
  A((i-1)*n0+1 : i*n0, (i-1)*n0+1 : i*n0) = A0;
end
A = A + A';
A(1:n+1:n^2) = 0;
A = sparse(double(A > 0));

% Output
% save mixture.mat A X Y idx_train idx_valid idx_test

% Visualization
figure(1);
for i = 1:c
  plot(Xc{i}(:,1), Xc{i}(:,2), '.');
  if i == 1
    hold on;
  elseif i == c
    hold off;
  end
end

figure(2);
spy(A);
