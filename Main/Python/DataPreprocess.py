import copy

import numpy as np


class DataPreprocess:
    def __init__(self, Dataset_input, **kwargs):
        self.kwargs = self.kwargs_construct(**kwargs)
        # Note, since every element in multi-graph list X has the same size and
        # node index, there will be only one column in Y for the node labels
        self.Y = Dataset_input.Y
        self.n = Dataset_input.n
        (self.X, self.Graph_count) = self.input_prep(Dataset_input.X)
        (self.attributes, self.U) = self.check_attributes()
        self.Dataset_input = Dataset_input


    def kwargs_construct(self, **kwargs):
        defaultKwargs = {'DiagA': True,'Laplacian': False,  #input_prep
                         'Correlation': True,      # graph_encoder_embed
                         'Attributes': False,      # GNN_preprocess
                         }
        kwargs = { **defaultKwargs, **kwargs}  # update the args using input_args
        return kwargs


    def check_attributes(self):
        """
          return attributes detected flag and attributes U
        """
        kwargs = self.kwargs

        Attributes_detected = False
        U = None

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
