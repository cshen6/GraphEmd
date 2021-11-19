## source(".../GraphEncoder.R") 
##suppressMessages(require(igraph))
suppressMessages(require(Matrix))
##suppressMessages(require(irlba))
suppressMessages(require(mclust))
##suppressMessages(require(gmmase))
suppressMessages(require(wordspace))

## Args:
##   X: either n*n adjacency matrix, or s*2 or s*3 (unweighted or weighted) edgelist
##   Y: size n*1 Class label vector, or a positive integer for number of classes. Known class labels shall be ordered from 1 to k. Unknown class label should be set to non-positive number (say 0 or -1).
##      When there is no known label, set Y to the desired K instead. 
##   Laplacian: True or False. Default to False uses adjacency matrix, while true uses the Laplacian transformation D^-0.5 * Adj * D^-0.5.
##   DiagA: whether to use diagonal augmentation or not. Default is true, and adds 1 to the diagonals of the adjacency matrix.
##   Correlation: whether the embedding is compared via correlation distance or not. Default is true.
##
## Return:
##   result: Encoder Embedding Z of size n*k, Y is the final label, indT is the index of known labels, W is the encoder projection matrix.
##
## Reference:
##   C. Shen and Q. Wang and C. E. Priebe, "Graph Encoder Embedding", 2021. 

GraphEncoder <- function(X, Y, Laplacian = FALSE, MaxIter=50, DiagA = TRUE, Correlation = TRUE) {
  
  s=dim(X);
  t=s[2];s=s[1];
  if (t<=3){
    n = max(max(X));
    if (t==2){
      X=cbind(X,matrix(1, nrow = s, ncol = 1));
    }
    if (DiagA==TRUE){
      XNew=cbind(1:n,1:n,1);
      X=rbind(X,XNew);
    }
  } else {
    n = dim(X)[1];
    if (DiagA==TRUE){
      X=X+diag(n);
    }
  }
  
  if (length(Y)==1){ #clustering
    K=Y;
    ariv = rep(0, MaxIter);
    Y2 = matrix(sample(K,n,rep=T), n, 1);
    for (i in 1:MaxIter) {
      result = GraphEncoderMain(X, Y2, Laplacian, Correlation);
      #mc = Mclust(result$Z, verbose = FALSE);
      #Y = mc$class;
      mc = kmeans(result$Z, K); 
      Y = mc$cluster;
      ariv[i] = adjustedRandIndex(Y2, Y)
      if (ariv[i] == 1) {
        stop()
      } else {
        Y2 = matrix(Y, n, 1)
      }
    }
  } else {
    result = GraphEncoderMain(X, Y,Laplacian,Correlation)
  }
  return(result)
}

GraphEncoderMain <- function(X, Y, Laplacian = FALSE, Correlation = TRUE) {
  n = dim(Y)[1];
  indT=(Y>0);
  Y1=Y[indT,];
  ##[tmp,~,Ytmp]=unique(Y1);
  tmp=unique(Y1);
  k=max(tmp);
  s=dim(X);
  t=s[2];s=s[1];
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
    if (t==3){
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
      D=D^-0.5;
      for (i in 1:s){
        X[i,3]=X[i,3]*D[X[i,1]]*D[X[i,2]];
      }
    } else {
      D=as.vector(pmax(rowSums(X),1)^-(0.5));
      for (i in 1:n){
        X[,i]=X[,i]*D[i] %*% D;
      }
    } 
  }
  
  ## matrix version
  if (s==n && t==n){
    Z=X%*%W;
  } 
  ## edgelist version
  if (t==3){
    Z=matrix(0, nrow = n, ncol = k);
    for (i in 1:s) {
      a=X[i,1];
      b=X[i,2];
      c=Y[a];
      d=Y[b];
      e=X[i,3];
      Z[a,d]=Z[a,d]+W[b,d]*e;
      if (a!=b){
          Z[b,c]=Z[b,c]+W[a,c]*e;
      }
    }
  }
  
  if (Correlation==TRUE){
    Z=normalize.rows(Z, method = "euclidean", p = 2);
    Z[is.na(Z)] = 0;
  }
  result = list(Z = Z, indT = indT, Y = Y, W = W)
  return(result)
}