function [S, Q] = louvain(A,Y)

if nargin<2
    Y=0;
end
k = full(sum(A));
twom = sum(k);
B = @(v) A(:,v) - k'*k(v)/twom;
if Y==0
    [S,Q] = genlouvain(B,10000,0,0,0);
else
    [S,Q] = genlouvain(B,10000,0,0,0,Y);
end
Q = Q/twom;
