import copy

import numpy as np
from numpy import linalg as LA
from numba import jit
# from tensorflow.keras.utils import to_categorical


class DataPreprocess:
    def __init__(self, Dataset_input, **kwargs):
        self.kwargs = self.kwargs_construct(**kwargs)
        # Note, since every element in multi-graph list X has the same size and
        # node index, there will be only one column in Y for the node labels
        self.Y = Dataset_input.Y
        self.n = Dataset_input.n
        (self.X, self.Graph_count) = self.input_prep(Dataset_input.X)
        (self.attributes, self.U) = self.check_attributes(Dataset_input)
        self.Dataset_input = Dataset_input


    def kwargs_construct(self, **kwargs):
        defaultKwargs = {'DiagA': True,'Laplacian': False,  #input_prep
                         'Correlation': True,      # graph_encoder_embed
                         'Attributes': False,      # GNN_preprocess
                         }
        kwargs = { **defaultKwargs, **kwargs}  # update the args using input_args
        return kwargs


    def check_attributes(self, Dataset_input):
        """
          return attributes detected flag and attributes U
        """
        kwargs = self.kwargs

        Attributes_detected = False
        U = None
        n = Dataset_input.n

        if kwargs["Attributes"]:
            U = kwargs["Attributes"]
            if U.shape[0] == n:
                Attributes_detected = True
            else:
                print("Attributes need to have the same size as the nodes.\
        If n nodes, need n rows")
        return Attributes_detected, U

    def supervise_preprocess(self):
        """
          adding test sets for supervised learning
          this function assumes only one test set
          if there is a list of test set, needs to modify this function
        """

        DataSets = self.semi_supervise_preprocess()
        Dataset_input = DataSets.Dataset_input

        DataSets.z_test = DataSets.Z[Dataset_input.test_idx]
        DataSets.Y_test = Dataset_input.Y_test.ravel()
        DataSets.z_unlabel = None
        DataSets.Y_unlabel = None

        return DataSets

    def test_edg_list_to_adj(self, n_test, n, edg_list):
        adj = np.zeros((n_test,n))

        for row in edg_list:
            [node_i, node_j, edge_i_j] = row
            adj[node_i, node_j] = edge_i_j

        return adj


    def input_prep(self, X):
        """
          X may be a single numpy object or a list of numpy objects
          The multi-graph input X is assumed has the same node numbers
          for each element in X, and the node are indexed the same way
          amonge the elements. e.g. node_0 in X[1] is the same node_0 in X[2].
          return X as a list of s*3 edge lists
          return n, which is the total number of nodes
        """

        # need total labeled number n
        # if try to get from the edg list, it may miss the node that has no connection with others but has label
        n = self.n

        # if X is a single numpy object, put this numpy object in a list
        if type(X) == np.ndarray:
            X = [X]

        ## Now X is a list of numpy objects
        # each element can be a numpy object for adjacency matrix or edge list

        Graph_count = len(X)

        for i in range(Graph_count):
            X_tmp = X[i]
            X_tmp = self.to_s3_list(X_tmp)

            X_tmp = self.single_X_prep(X_tmp, n)
            X[i] = X_tmp

        return X, Graph_count


    def to_s3_list(self,X):
        """
          the input X is a signle graph, can be adjacency matrix or edgelist
          this function will return a s3 edge list
        """
        (s,t) = X.shape

        if s == t:
            # convert adjacency matrix to edgelist
            X = self.adj_to_edg(X);
        else:
            # for either s*2 or s*3 case, calculate n -- vertex number
            if t == 2:
                # enlarge the edgelist to s*3 by adding 1 to the thrid position as adj(i,j)
                X = np.insert(X, 1, np.ones(s,1))
        return X


    def single_X_prep(self, X, n):
        """
          input X is a single S3 edge list
          this adds Diagnal augement and Laplacian normalization to the edge list
        """
        kwargs = self.kwargs

        # Diagnal augment
        if kwargs['DiagA']:
            # add self-loop to edg list -- add 1 connection for each (i,i)
            self_loops = np.column_stack((np.arange(n), np.arange(n), np.ones(n)))
            # faster than vstack --  adding the second to the bottom
            X = np.concatenate((X,self_loops), axis = 0)

        # Laplacian
        s = X.shape[0] # get the row number of the edg list
        if kwargs["Laplacian"]:
            D = np.zeros((n,1))
            for row in X:
                [v_i, v_j, edg_i_j] = row
                v_i = int(v_i)
                v_j = int(v_j)
                D[v_i] = D[v_i] + edg_i_j
                if v_i != v_j:
                    D[v_j] = D[v_j] + edg_i_j

            D = np.power(D, -0.5)

            for i in range(s):
                X[i,2] = X[i,2] * D[int(X[i,0])] * D[int(X[i,1])]

        return X

    def adj_to_edg(self,A):
        """
          input is the adjacency matrix: A
          other variables in this function:
          s: number of edges
          return edg_list -- matrix format with shape(edg_sum,3):
          example row in edg_list(matrix): [vertex1, vertex2, connection weight from Adj matrix]
        """
        # check the len of the second dimenson of A
        if A.shape[1] <= 3:
            edg = A
        else:
            n = A.shape[0]
            # construct the initial edgg_list matrix with the size of (edg_sum, 3)
            edg_list = []
            for i in range(n):
                for j in range(n):
                    if A[i,j] > 0:
                        edg_list.append([i, j, A[i,j]])
            edg = np.array(edg_list)
        return edg

    def semi_supervise_preprocess(self):
        """
          get Z, W using multi_graph_encoder_embed()
          get training sets and testing sets for Z and Y by using split_data()

        """
        DataSets =  copy.deepcopy(self)
        Y = DataSets.Y

        (Z, W) = multi_graph_encoder_embed(DataSets, Y)

        DataSets.Z = Z
        DataSets.W = W
        DataSets.k = DataSets.get_k()
        DataSets = DataSets.split_data()

        return DataSets


    def get_k(self):
        Y = self.Y
        n = self.n
        # get class number k or the largest cluster size
        # max of all flattened element + 1
        if len(Y) == n:
            k = np.amax(Y) + 1
        return k


    def split_data(self):
        split_Sets =  copy.deepcopy(self)

        Y = split_Sets.Y
        Z = split_Sets.Z

        ind_train = np.argwhere (Y >= 0)[:,0]
        ind_unlabel = np.argwhere (Y < 0)[:,0]

        Y_train = Y[ind_train, 0]
        z_train = Z[ind_train]

        Y_unlabel = None
        z_unlabel = None

        if len(ind_unlabel) > 0:
            Y_unlabel = Y[ind_unlabel, 0]
            z_unlabel = Z[ind_unlabel]

            # Convert targets into one-hot encoded format
        Y_train_one_hot = to_categorical(Y_train)

        split_Sets.ind_unlabel = ind_unlabel
        split_Sets.ind_train = ind_train
        split_Sets.Y_train = Y_train
        split_Sets.Y_unlabel = Y_unlabel
        split_Sets.z_train = z_train
        split_Sets.z_unlabel = z_unlabel
        split_Sets.Y_train_one_hot = Y_train_one_hot

        return split_Sets


    def DataSets_reset(self, option):
        """
          based on the information of the given new Y:
          1. reassign Z and W to the given DataSets,
          2. update z_train, z_unlabel
          Input Option:
          1. if the option is "y_temp", do graph encoder using y_temp
        """
        NewSets =  copy.deepcopy(self)
        kwargs = NewSets.kwargs
        ind_unlabel = NewSets.ind_unlabel
        ind_train = NewSets.ind_train
        y_temp =  NewSets.y_temp


        # different versions
        if option == "y_temp":
            [Z,W] = multi_graph_encoder_embed(NewSets, y_temp)
        if option == "y_temp_one_hot":
            y_temp_one_hot = NewSets.y_temp_one_hot
            [Z,W] = multi_graph_encoder_embed(NewSets, y_temp_one_hot)
        if NewSets.attributes:
            # add U to Z side by side
            Z = np.concatenate((Z, NewSets.U), axis=1)

        NewSets.Z = Z
        NewSets.z_train = Z[ind_train]
        NewSets.z_unlabel = Z[ind_unlabel]

        return NewSets


@jit(nopython=True)
def X_prep_laplacian(X, n):
    """
      input X is a single S3 edge list
      this adds Diagnal augement and Laplacian normalization to the edge list
      Taken from DataPreprocesss.single_X_prep()
    """

    s = X.shape[0] # get the row number of the edg list
    # if kwargs["Laplacian"]:
    D = np.zeros((n,1), dtype=np.int32)
    for row in X: # Iterate over edges
        # TODO What about self-edges at the end?
        [v_i, v_j, edg_i_j] = row
        v_i = int(v_i)
        v_j = int(v_j)
        edg_i_j = int(edg_i_j)

        D[v_i] += edg_i_j
        if v_i != v_j: # Only fails for self-edges
            D[v_j] += edg_i_j
    # In Ligra, the above is calculated for us, and is present in v[i].getInDegree()/getOutDegree()

    D = np.power(D, -0.5)

    for i in range(s):
        X[i,2] *= (D[int(X[i,0])] * D[int(X[i,1])])[0] # Turns from ndarray of 1 element to float

    return X


@jit(nopython=True)
def numba_main_embedding(X, Y, W, possibility_detected, n, k):
    # Edge List Version in O(s)
    Z = np.zeros((n,k))
    i = 0


    for row in X: # Loop over edges once only?
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

            if label_j >= 0: # Why > 0 label?
                Z[v_i, label_j] += W[v_j, label_j]*edg_i_j
            if (label_i >= 0) and (v_i != v_j):
                Z[v_j, label_i] += W[v_i, label_i]*edg_i_j

    return Z


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

    if kwargs["Laplacian"]:
        X = X_prep_laplacian(X, n)

    if possibility_detected:
        # sum Y (each row of Y is a vector of posibility for each class), then do element divid nk.
        # Ariel: I think this is the Laplacian part
        nk=np.sum(Y, axis=0)
        W=Y/nk
    else:
        for i in range(k):
            nk[0,i] = np.count_nonzero(Y[:,0]==i)

        for i in range(Y.shape[0]): # Y.shape[0] == n_vertices
            k_i = Y[i,0]
            if k_i >=0:
                W[i,k_i] = 1/nk[0,k_i]

    Z = numba_main_embedding(X, Y, W, possibility_detected, n, k)

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
