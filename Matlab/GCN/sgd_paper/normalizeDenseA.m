function A = normalizeDenseA(A)
% Normalize dense A.

n = size(A,1);
A = A + eye(n);
sqrt_diagD = sqrt(sum(A,2));
A = A ./ (sqrt_diagD * sqrt_diagD');
