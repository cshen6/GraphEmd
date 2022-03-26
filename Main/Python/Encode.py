import numpy as np
from numpy import linalg as LA



############------------graph_encoder_embed_start----------------###############
def graph_encoder_embed(X,Y,n,**kwargs):
    """
      input X is s*3 edg list: nodei, nodej, connection weight(i,j)
      graph embedding function
    """
    defaultKwargs = {'Correlation': True}
    kwargs = { **defaultKwargs, **kwargs}

    #If Y has more than one dimention , Y is the range of cluster size for a vertex. e.g. [2,10], [2,5,6]
    # check if Y is the possibility version. e.g.Y: n*k each row list the possibility for each class[0.9, 0.1, 0, ......]
    possibility_detected = False
    if Y.shape[1] > 1:
        k = Y.shape[1]
        possibility_detected = True
    else:
        # assign k to the max along the first column
        # Note for python, label Y starts from 0. Python index starts from 0. thus size k should be max + 1
        k = Y[:,0].max() + 1

    #nk: 1*n array, contains the number of observations in each class
    #W: encoder marix. W[i,k] = {1/nk if Yi==k, otherwise 0}
    nk = np.zeros((1,k))
    W = np.zeros((n,k))

    if possibility_detected:
        # sum Y (each row of Y is a vector of posibility for each class), then do element divid nk.
        nk=np.sum(Y, axis=0)
        W=Y/nk
    else:
        for i in range(k):
            nk[0,i] = np.count_nonzero(Y[:,0]==i)

        for i in range(Y.shape[0]):
            k_i = Y[i,0]
            if k_i >=0:
                W[i,k_i] = 1/nk[0,k_i]

    # Edge List Version in O(s)
    Z = np.zeros((n,k))
    i = 0
    for row in X:
        [v_i, v_j, edg_i_j] = row
        v_i = int(v_i)
        v_j = int(v_j)
        if possibility_detected:
            for label_j in range(k):
                Z[v_i, label_j] = Z[v_i, label_j] + W[v_j, label_j]*edg_i_j
                if v_i != v_j:
                    Z[v_j, label_j] = Z[v_j, label_j] + W[v_i, label_j]*edg_i_j
        else:
            label_i = Y[v_i][0]
            label_j = Y[v_j][0]

            if label_j >= 0:
                Z[v_i, label_j] = Z[v_i, label_j] + W[v_j, label_j]*edg_i_j
            if (label_i >= 0) and (v_i != v_j):
                Z[v_j, label_i] = Z[v_j, label_i] + W[v_i, label_i]*edg_i_j

    # Calculate each row's 2-norm (Euclidean distance).
    # e.g.row_x: [ele_i,ele_j,ele_k]. norm2 = sqr(sum(2^2+1^2+4^2))
    # then divide each element by their row norm
    # e.g. [ele_i/norm2,ele_j/norm2,ele_k/norm2]
    if kwargs['Correlation']:
        row_norm = LA.norm(Z, axis = 1)
        reshape_row_norm = np.reshape(row_norm, (n,1))
        Z = np.nan_to_num(Z/reshape_row_norm)

    return Z, W

def multi_graph_encoder_embed(DataSets, Y):
    """
      input X contains a list of s3 edge list
      get Z and W by using graph emcode embedding
      Z is the concatenated embedding matrix from multiple graphs
      if there are attirbutes provided, add attributes to Z
      W is a list of weight matrix Wi
    """

    X = DataSets.X
    n = DataSets.n
    U = DataSets.U
    Graph_count = DataSets.Graph_count
    attributes = DataSets.attributes
    kwargs = DataSets.kwargs

    W = []

    for i in range(Graph_count):
        if i == 0:
            [Z, Wi] = graph_encoder_embed(X[i],Y,n,**kwargs)
        else:
            [Z_new, Wi] = graph_encoder_embed(X[i],Y,n,**kwargs)
            Z = np.concatenate((Z, Z_new), axis=1)
        W.append(Wi)

    # if there is attributes matrix U provided, add U
    if attributes:
        # add U to Z side by side
        Z = np.concatenate((Z, U), axis=1)

    return Z, W


class Encoder_case:
    def __init__(self, A,Y,n):
        Encoder_case.X = A
        Encoder_case.Y = Y
        Encoder_case.n = n


if __name__ == '__main__':
    A = np.ones((5,5))
    A[0,4] = 0
    A[4,0] = 0
    np.fill_diagonal(A, 0)

    Y = np.array([[0,0,0,1,1]]).reshape((5,1))

    print(A)
    print(Y)

    Encoder_case5 = Encoder_case(A,Y,5)

    from Main.Python.DataPreprocess import DataPreprocess

    Dataset = DataPreprocess(Encoder_case5, Laplacian = False, DiagA = False)
    print(Dataset.X)
    print(Dataset.Y)
    print(Dataset.n)

    Z, W = graph_encoder_embed(Dataset.X[0], Dataset.Y, Dataset.n, Correlation = False)
    print(Z)
    print(W)
