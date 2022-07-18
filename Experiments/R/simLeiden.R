

suppressMessages(require(igraph))
#suppressMessages(require(leiden))
suppressMessages(require(leidenAlg))
suppressMessages(require(R.matlab))


Z=readMat("C:/Work/Applications/GitHub/Phone.mat");
Edge=Z$Edge;
A=graph_from_edgelist(Edge[,1:2]);

#Z=cluster_leiden(A,weights=Edge[,3]);
YL=leiden.community(A,resolution=1)$membership;
#YL=rleiden.community(A,min.community.size=2, n.cores=1);

#Y=leiden(A,weights=Edge[,3],resolution_parameter=10,n_iterations=5);
#write.csv(YL,"C:/Work/Applications/GitHub/GraphNN/Data/Matlab/tmpL.csv", row.names = FALSE)


Z=readMat("C:/Work/Applications/Github/GraphNN/Data/Matlab/lastfm.mat");
Edge=Z$Edge;
A=graph_from_edgelist(Edge[,1:2]);
Y1=leiden.community(A,resolution=0.1)$membership;
Y2=leiden.community(A,resolution=0.5)$membership;
Y3=leiden.community(A,resolution=1)$membership;
Y4=leiden.community(A,resolution=2)$membership;
Y5=leiden.community(A,resolution=5)$membership;
Y6=leiden.community(A,resolution=10)$membership;
YL=cbind(Y1,Y2,Y3,Y4,Y5,Y6);
write.csv(YL,"C:/Work/Applications/GitHub/GraphNN/Data/Matlab/YL.csv", row.names = FALSE)
