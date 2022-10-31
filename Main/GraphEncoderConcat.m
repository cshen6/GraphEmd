function [Z]=GraphEncoderConcat(Adj,Y,level,concat)

if nargin<3
    level=size(Y,2);
end
if nargin<4
    concat=1;
end
if level>size(Y,2)
    level=size(Y,2);
end

[Z]=GraphEncoder(Adj,Y(:,1));
d=size(Z,2);
for i=2:level
    [tmpZ]=GraphEncoder(Adj,Y(:,i));
     [~,tmpZ] = pca(tmpZ,'NumComponents',d);
    if concat==0
        Z=Z+tmpZ;
    else
        Z=[Z,tmpZ];
    end
end