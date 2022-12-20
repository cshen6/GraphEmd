function [Z]=GraphEncoderConcat(Adj,Y,level,concat,directed)
if nargin<3
    level=size(Y,2);
end
if nargin<4
    concat=1;
end
if nargin<5
    directed=2;
end
if level>size(Y,2)
    level=size(Y,2);
end

opts = struct('Directed',directed,'Normalize',true);
[Z]=GraphEncoder(Adj,Y(:,1),opts);
d=size(Z,2);
for i=2:level
    [tmpZ]=GraphEncoder(Adj,Y(:,i),opts);
     [~,tmpZ] = pca(tmpZ,'NumComponents',d);
    if concat==0
        Z=Z+tmpZ;
    else
        Z=[Z,tmpZ];
    end
end