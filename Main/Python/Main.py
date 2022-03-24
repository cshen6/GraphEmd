import numpy as np
from numpy import linalg as LA

# Supress/hide the warning
# invalide devide resutls will be nan
from DataPreprocess import DataPreprocess
from Main.Python.Clustering import Clustering
from Main.Python.Evaluation import Evaluation
from Main.Python.GNN import GNN
from Main.Python.LDA import LDA

np.seterr(invalid='ignore')
############------------Auto_select_method_start-----------------###############
def Run(case, opt, **kwargs):
    """
      input X can be a list of one of these format below:
      1. python list of n*n adjacency matrices.
      2. python list of s*2 edge lists.
      3. python list of s*3 edge lists.
      input Y can be these choices below:
      1. no Y input. The default will be [2,3,4,5] -- K range for clusters.
      2. n*1 class -- label vector. Positive labels are knwon labels and -1 indicate unknown labels.
      3. A range of potential number of clusters -- K (K clusters in total), i.e., [3, 4, 5].

      if input X is n*n adjacency =>  s*3 edg list
      if input X is s*2 => s*3 edg list

      Vertex size should be >10.

      Clustering / Classification
      The program automaticlly decide to run clustering or classification.
      1. If Y is a given cluster range, do clustering (case 1,3 for Y).
      2. If Y is a label vector (case 2 for Y), do classification.
      For classification: semi-supervised learning, supervised learning methods.
                          see the "Learner" defined below.


      Supervised learning "Learner":
        **Note the input trining set (X) need has fully known labels in Y.
        Learner = 1 run LDA, test on test set
        Learner = 2 run NN, test on test set

      Semi-supervised learning "Learner":
         **Note the input trining set (X) need some unknown label(s) in Y.
        Learner=0 means embedding via known label, do not learn the unknown labels.
             Since only some nodes in the training set has known label,
             the test set is the unknwon labeled set, which is compared with
             the original labels of the unknown set
        Learner=1 means embedding via partial label, then learn unknown label via LDA.
          this runs semi-supervised learning with NN,
          the test will be on the result labels with the original labels

        Learner=2 means embedding via partial label, then learn unknown label via two-layer NN.
          this runs semi-supervised learning with NN,
          the test will be on the result labels with the original labels


    """
    defaultKwargs = {'Y':[2,3,4,5], 'DiagA': True,'Correlation': True,'Laplacian': False,
                     'Learner': 1, 'LearnerIter': False, 'MaxIter': 50, 'MaxIterK': 5,
                     'Replicates': 3, 'Attributes': False, 'neuron': 20, 'activation': 'relu'}
    kwargs = { **defaultKwargs, **kwargs }

    eval = Evaluation()
    kwargs_for_DataPreprocess =  {k: kwargs[k] for k in ['DiagA', 'Laplacian', 'Correlation', 'Attributes']}
    Dataset = DataPreprocess(case, **kwargs_for_DataPreprocess)

    Y = case.Y
    n = case.n

    # auto check block
    # if the option is not clustering, but the Y does not contain labels (known/unknwon) for n nodes.
    if (opt != "c") and (len(Y) != n):
        opt = "c" # do clustering
        print("The given Y do not have the same size as the node.Y is assumed as cluster number range.",
              "Clustering will be performed.",
              "If you want to do classification, stop the current run, reimport the Y with the right format then run again.",
              sep = "\n")


    # clustering
    if opt == 'c':
        cluster = Clustering(Dataset)
        Z, Y, W, meanSS = cluster.cluster_main()
        ari = eval.clustering_test(Y, case.Y_ori)
        print("ARI: ", ari)

    # supervised learning
    if opt == "su":
        Dataset = Dataset.supervise_preprocess()
        kwargs_for_learner = {k: kwargs[k] for k in ['Learner', 'LearnerIter']}
        if kwargs['Learner'] == 1:
            lda = LDA(Dataset, **kwargs_for_learner)
            lda_res = lda.LDA_Learner()
            acc = eval.LDA_supervise_test(lda_res, Dataset.z_test, Dataset.Y_test)
        if kwargs['Learner'] == 0:
            gnn = GNN(Dataset, **kwargs_for_learner)
            gnn_res = gnn.GNN_complete()
            acc = eval.GNN_supervise_test(gnn_res, Dataset.z_test, Dataset.Y_test)
        print("acc: ", acc)

    # semisupervised learning
    if opt == "se":
        Dataset = Dataset.semi_supervise_preprocess()
        kwargs_for_learner = {k: kwargs[k] for k in ['Learner', 'LearnerIter']}
        if kwargs['Learner'] == 2:
            gnn = GNN(Dataset, **kwargs_for_learner)
            gnn_res = gnn.GNN_complete()
            acc = eval.GNN_semi_supervised_learn_test(gnn_res.Y, case.Y_ori)
        if kwargs['Learner'] == 1:
            lda = LDA(Dataset, **kwargs_for_learner)
            lda_res = lda.LDA_Iter()
            acc = eval.GNN_semi_supervised_learn_test(lda_res.Y, case.Y_ori)
        if kwargs['Learner'] == 0:
            gnn = GNN(Dataset, **kwargs_for_learner)
            gnn_res = gnn.GNN_complete()
            acc = eval.GNN_semi_supervised_not_learn_test(gnn_res, Dataset, case)
        print("acc: ", acc)



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

