function plotUMAP(X,Y)
map2 = brewermap(128,'PiYG'); % brewmap
colormap(gca,map2);
K=max(Y);
hold on
for k=1:K
    tmp=(Y==k);
    plot(X(tmp,1),X(tmp,2),'.','MarkerSize',2)
end
hold off