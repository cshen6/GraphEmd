function W = initW(sz)
% Glorot initialization

W = (2*rand(sz) - 1) * sqrt(6/sum(sz));
