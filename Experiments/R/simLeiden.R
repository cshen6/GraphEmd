

suppressMessages(require(igraph))
suppressMessages(require(leiden))
suppressMessages(require(R.matlab))


Z=readMat("../Phone.mat");
Edge=Z$Edge;
A=graph_from_edgelist(Edge[,1:2]);

#Z=cluster_leiden(A,weights=Edge[,3]);

Y=leiden(A,weights=Edge[,3],resolution_parameter=10,n_iterations=5);
write.csv(Y1,"../PhoneL.csv", row.names = FALSE)


Z=readMat("../Github/GraphNN/Data/Matlab/lastfm.mat");
Edge=Z$Edge;
A=graph_from_edgelist(Edge[,1:2]);
Y=leiden(A,resolution_parameter=10,n_iterations=5);

