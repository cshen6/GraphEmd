function A = normalizeSparseA(A)
% Normalize sparse A.

n = size(A,1);
A = A + speye(n);
sqrt_diagD = sqrt(full(sum(A,2)));

[ii, jj, vv] = find(A);
vv = vv ./ (sqrt_diagD(ii) .* sqrt_diagD(jj));

A = sparse(ii, jj, vv, n, n);
