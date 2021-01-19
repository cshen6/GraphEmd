function [label]=GraphNNPredict(tsn,mdl,filter,M,option)

% the testing size needs to match the filter size
if nargin<4
    M=0;
end
if nargin<5
    option=2;
end
% if size(tsn,2)~=size(filter,1)
%     return
% end
if option==2
    [n,k]=size(filter);
    m=size(tsn,2);
    for i=1:m/n
       tsn(:,(i-1)*k+1:i*k)=tsn(:,(i-1)*n+1:i*n)*filter;
    end
end
if size(M,1)==size(tsn,1)
    tsn=[tsn,M];
end
label = mdl(tsn');
% label=predict(mdl,tsn*filter);
% label = vec2ind(label);