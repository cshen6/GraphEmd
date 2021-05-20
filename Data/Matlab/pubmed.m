labels=table2array(PubMednodelabels);
edges=table2array(PubMededges);
[a,b,~]=unique(labels(:,1));
Y=[b,labels(:,2)];
Y=sort(Y,1);
Y=Y(:,2);
X=edges;
for i=1:max(b)
    tmp=(X==a(i));
    X(tmp)=i;
end


[Adj,Y]=edge2adj(X,Y);