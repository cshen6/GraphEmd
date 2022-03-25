import numpy as np


# Supress/hide the warning
# invalide devide resutls will be nan
from Main.Python.DataPreprocess import DataPreprocess, graph_encoder_embed
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


if __name__ == '__main__':
    gimi = graph_encoder_embed()
