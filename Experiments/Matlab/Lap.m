function Adj=Lap(Adj)

n=size(Adj,1);
% D=mean(Adj,2)^0.5;
D=max(sum(Adj,1),1).^(0.5);
for j=1:n
    Adj(:,j)=Adj(:,j)/D(j)./D';
end