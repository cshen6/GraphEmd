function X = normalizeX(X)
% Row normalize X.

%no = sqrt(sum(X.^2,2));
no = sum(X,2); % Kipf's way of normalization
no(no == 0) = 1;
X = X ./ no;
