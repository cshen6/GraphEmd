## source(".../GraphEncoder.R") 
suppressMessages(require(igraph))
suppressMessages(require(Matrix))
suppressMessages(require(irlba))
suppressMessages(require(mclust))
suppressMessages(require(gmmase))


GraphEncoder <- function(X, Y, Laplacian = FALSE) {
  ## Args:
  ##   X: either n*n adjacency matrix, or s*2 or s*3 (unweighted or weighted) edgelist
  ##   Y: size n*1 Class label vector, or a positive integer for number of classes. Known class labels shall be ordered from 1 to k. Unknown class label should be set to non-positive number (say 0 or -1).
  ##      When there is no known label, set Y to the desired K instead. 
  ##   Laplacian: True or False. Default to False uses adjacency matrix, while true uses the Laplacian transformation D^-0.5 * Adj * D^-0.5.
  ##
  ## Return:
  ##   result: Encoder Embedding Z of size n*k, indT is the index of known labels, Y is the re-ordered labels, W is the encoder projection matrix.
  ##
  ## Reference:
  ##   C. Shen and Q. Wang and C. E. Priebe, "Graph Encoder Embedding", 2021. 
  
  n = dim(Y)[1];
  indT=(Y>0);
  Y1=Y[indT,];
  ##[tmp,~,Ytmp]=unique(Y1);
  tmp=unique(Y1);
  k=max(tmp);
  s=dim(X);
  t=s[2];s=s[1];
  if (t==2){
    X=cbind(X,matrix(1, nrow = s, ncol = 1));
  }
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
    if (t<=3){
      D=array(0, dim = c(n, 1));
      for (i in 1:s){
        D(X[i,1])=D(X[i,1])+X[i,3];
        D(X[i,2])=D(X[i,2])+X[i,3];
      }
      D=D^-0.5;
      for (i in 1:s){
        X[i,3]=X[i,3]*D(X[i,1])*D(X[i,2]);
      }
    } else {
      D=diag(as.vector(max(rowSums(Adj),1)^-(0.5)),n,n);
      X=D%*%X%*%D;
    }
  }
  
  if (s==n && t==n){
    Z=X%*%W;
  } else {
    Z=matrix(0, nrow = n, ncol = k);
    for (i in 1:s) {
      a=X[i,1];
      b=X[i,2];
      c=Y[a];
      d=Y[b];
      e=X[i,3];
      Z[a,d]=Z[a,d]+W[b,d]*e;
      Z[b,c]=Z[b,c]+W[a,c]*e;
    }
  }
  
  result = list(Z = Z, indT = indT, Y = Y, W = W)
  return(result)
}