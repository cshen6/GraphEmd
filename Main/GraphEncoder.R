## source(".../GraphEncoder.R") 
suppressMessages(require(igraph))
suppressMessages(require(Matrix))
##suppressMessages(require(irlba))
suppressMessages(require(mclust))
##suppressMessages(require(gmmase))
suppressMessages(require(wordspace))
options(warn =-1) 

## Args:
##   X: either n*n adjacency matrix, or s*2 or s*3 (unweighted or weighted) edgelist. Vertex size should be >10.
##   Y: size n*1 Class label vector, or a positive integer for number of classes, or a range of potential cluster size, i.e., [2,10]. 
##      In case of partial known labels, Y should be n*1 with unknown class label set to non-positive number (say 0 or -1).
##      When there is no known label, set Y to the desired K, or a range of potential K. 
##   Laplacian specifies whether to uses graph Laplacian or adjacency matrix; 
##   Correlation specifies whether to use angle metric or Euclidean metric;
##   DiagA = true augment all diagonal entries by +1 (i.e., add self-loop to edgelist);
##   Three options for clustering: Replicates denotes the number of replicates for clustering, 
##                               MaxIter denotes the max iteration within each replicate for encoder embedding,
##                               MaxIterK denotes the max iteration used within kmeans.
##
## Return:
##   result: Encoder Embedding Z of size n*k, Y is the final label, W is the encoder projection matrix. 
##           For clustering, also output meanSS as the total within-cluster sum of squares divided by total sum of squares then normalized by K/2; 
##           For classification, also output indT as the index of known labels.
##
## Reference:
##   C. Shen and Q. Wang and C. E. Priebe, "Graph Encoder Embedding", 2021. 
##   C. Shen et.al., "Graph Encoder Clustering", in preparation.

## Main Function
## X=as.matrix(get.data.frame(g));
GraphEncoder <- function(X, Y=c(2:5), Laplacian = FALSE, DiagA = TRUE, Correlation = TRUE, MaxIter=50, MaxIterK=5, Replicates=5) {
  
  s=dim(X);
  t=s[2];s=s[1];
  
  if (s==t){ #convert adjacency matrix to edgelist
    X = graph.adjacency(X,weighted=TRUE);
    n=vcount(X);
    X = as.matrix(get.data.frame(X));
  }
  if (t==2){
    X=cbind(X,matrix(1, nrow = s, ncol = 1));
  } 
  n = max(max(X[,1:2]))[1];
  if (DiagA==TRUE){
    XNew=cbind(1:n,1:n,1);
    X=rbind(X,XNew);
  }
  
  ## partial or full known labels when label size matches vertex size, do embedding / classification directly
  if (length(Y)==n){
    result = GraphEncoderEmbed(X, Y, n, Laplacian,Correlation);
    meanSS=0;Y=result$Y;indT=result$indT;
  } else {
    ## otherwise do clustering
    indT=matrix(0, nrow = n, ncol = 1);
    K=Y;
    ## when a given cluster size is specified
    if (length(K)==1){ #clustering
      result=GraphEncoderCluster(X,K,n,Laplacian,Correlation, MaxIter, MaxIterK, Replicates);
      Y=result$Y;meanSS=result$minSS;
    } else {
      ## when a range of cluster size is specified
      if (length(K)<n/2 & max(K)<max(n/2,10)){
        minSS=-1;Z=0;W=0;meanSS = rep(0, length(K));
        for (r in 1:length(K)){
          resTmp = GraphEncoderCluster(X,K[r],n,Laplacian,Correlation, MaxIter, MaxIterK, Replicates);
          tmp=resTmp$minSS;
          meanSS[r]=tmp;
          if (minSS==-1 | tmp<minSS){
            minSS=tmp;
            result=resTmp;
          }
        }
        Y=result$Y;
      }
    }
  }
  result = list(Z = result$Z, Y = Y, W = result$W, meanSS=meanSS, indT=indT);
  return(result)
}

## Clustering Function
GraphEncoderCluster <- function(X, K, n, Laplacian = FALSE, Correlation = TRUE, MaxIter=50, MaxIterK=5, Replicates=3) {
  minSS = -1;
  #ariv = rep(0, MaxIter);
  for (rep in 1:Replicates){
    Y2 = matrix(sample(K,n,rep=T), n, 1);
    for (r in 1:MaxIter) {
      resTmp = GraphEncoderEmbed(X, Y2, n, Laplacian, Correlation);
      grp = kmeans(resTmp$Z, K, iter.max = MaxIterK); 
      Y3 = grp$cluster;
      #mc = Mclust(restmp$Z, verbose = FALSE);
      #Y3 = mc$class;
      #ariv[i] = adjustedRandIndex(Y2, Y3);
      if (adjustedRandIndex(Y2, Y3) == 1) {
        break
      } else {
        Y2 = matrix(Y3, n, 1);
      }
    }
    #tmp = max(mc$withinss) / mc$totss * K;
    tmp=grp$withinss^2; #within-cluster squared distance per cluster
    tmpBetween=matrix(0, nrow = K, ncol = 1); #between-cluster squared distance from each observation to the each center
    tmpCount=grp$size; #number of points per cluster
    for (i in 1:K){
      tmpBetween[i]=sum(colSums(resTmp$Z-matrix(grp$centers[i], nrow=n, ncol=K, byrow=TRUE))^2);
    }
    tmp=tmp/tmpCount/(tmpBetween-tmp)*(n-tmpCount)*tmpCount/n;
    tmp=mean(tmp)+2*var(tmp)^0.5;
    if (minSS==-1 | tmp<minSS){
      minSS=tmp;
      Y=Y3;
      result=resTmp;
    }
  }
  result = list(Z = result$Z, Y = Y, W = result$W, minSS=minSS);
  return(result)
}

## Embedding Function
GraphEncoderEmbed <- function(X, Y, n, Laplacian = FALSE, Correlation = TRUE) {
  indT=(Y>0);
  Y1=Y[indT,];
  s=dim(X)[1];
  tmp=unique(Y1);
  k=max(tmp);
  ##X=as.matrix(X,s,t);
  ##Y=as.matrix(Y,n,1);
  ##Y[indT]=Ytmp;
  
  nk=matrix(0, nrow = 1, ncol = k);
  W=matrix(0, nrow = n, ncol = k);
  indS=matrix(0, nrow = n, ncol = k);
  
  for (i in 1:k) {
    ind=(Y==i);
    nk[i]=sum(ind);
    W[ind,i]=1/nk[i];
    indS[,i]=ind;
  }
  
  if (Laplacian==TRUE){
    D=array(0, dim = c(n, 1));
    for (i in 1:s){
      a=X[i,1];
      b=X[i,2];
      c=X[i,3];
      D[a]=D[a]+c;
      if (a!=b){
        D[b]=D[b]+c;
      }
    }
    D=D.^-0.5;
    for (i in 1:s){
      X[i,3]=X[i,3]*D[X[i,1]]*D[X[i,2]];
    }
  }
  
  Z=matrix(0, nrow = n, ncol = k);
  for (i in 1:s) {
    a=X[i,1];
    b=X[i,2];
    c=Y[a];
    d=Y[b];
    e=X[i,3];
    if (d > 0){
      Z[a,d]=Z[a,d]+W[b,d]*e;
    }
    if (c > 0 & a!=b){
      Z[b,c]=Z[b,c]+W[a,c]*e;
    }
  }
  
  if (Correlation==TRUE){
    Z=normalize.rows(Z, method = "euclidean", p = 2);
    Z[is.na(Z)] = 0;
  }
  result = list(Z = Z, Y = Y, W = W, indT = indT)
  return(result)
}