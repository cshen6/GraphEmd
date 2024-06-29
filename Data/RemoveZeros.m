
function [Adj,Y]=RemoveZeros(AdjOri,YOri);

n=length(YOri);
ind=ones(n,1);
for i=1:n
    if norm(AdjOri(i,:))==0
        ind(i)=0;
    end
    AdjOri(i,i)=0;
end
ind=(ind>0);
Adj=AdjOri(ind,ind);
Y=YOri(ind);


% % Change integer naming to cell format
% numVariables = 18;
% G = cell(1, numVariables);
% for i = 1:numVariables
% variableName = sprintf('G%d', i);
% G{i} = eval(variableName);  % Store the actual variable (G1, G2, ..., G32)
% end