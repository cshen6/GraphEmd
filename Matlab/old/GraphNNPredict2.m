function [label]=GraphNNPredict2(tsn,mdl,filter,M,option)

% the testing size needs to match the filter size
if nargin<4
    M=0;
end
if nargin<5
    option=2;
end
if size(tsn,2)~=size(filter,1)
    return
end
if option==2
    tsn=tsn*filter;
end
if size(M,1)==size(tsn,1)
    tsn=[tsn,M];
end
label = mdl(tsn');
% label=predict(mdl,tsn*filter);
% label = vec2ind(label);