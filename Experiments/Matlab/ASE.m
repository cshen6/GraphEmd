function Z=ASE(Adj,d)

if nargin<2
    d=3;
end

[U,S,~]=svds(Adj,d);
Z=U(:,1:d)*S(1:d,1:d)^0.5;