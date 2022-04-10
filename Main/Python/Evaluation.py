from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from tensorflow.keras.utils import to_categorical
import numpy as np
from Main.Python.DataPreprocess import graph_encoder_embed

class Evaluation:
    def GNN_supervise_test(self, gnn, z_test, y_test):
        """
          test the accuracy for GNN_direct
        """
        y_test_one_hot = to_categorical(y_test)
        # set verbose to 0 to silent the output
        test_loss, test_acc = gnn.model.evaluate(z_test,  y_test_one_hot, verbose=0)

        return test_acc

    def LDA_supervise_test(self, lda, z_test, y_test):
        """
          test the accuracy for LDA_learner
        """
        test_acc = lda.model.score(z_test, y_test)

        return test_acc

    def GNN_semi_supervised_learn_test(self,Y_result, Y_original):
        """
          test accuracy for semi-supervised learning
        """
        test_acc = metrics.accuracy_score(Y_result, Y_original)

        return test_acc

    def GNN_semi_supervised_not_learn_test(self, gnn, Dataset, case):
        """
          test accuracy for semi-supervised learning
        """

        ind_unlabel = Dataset.ind_unlabel
        z_unlabel =  Dataset.z_unlabel
        y_unlabel_ori = case.Y_ori[ind_unlabel, 0]
        y_unlabel_ori_one_hot = to_categorical(y_unlabel_ori)
        test_loss, test_acc = gnn.model.evaluate(z_unlabel, y_unlabel_ori_one_hot, verbose=0)

        return test_acc


    def clustering_test(self, Y_result, Y_original):
        """
          test accuracy for semi-supervised learning
        """
        ari = adjusted_rand_score(Y_result, Y_original.reshape(-1,))

        return ari


# Code to test functions
class Encoder_case:
    def __init__(self, A,Y,n):
        Encoder_case.X = A
        Encoder_case.Y = Y
        Encoder_case.n = n


if __name__ == '__main__':
    # A = np.ones((5,5))
    # A[0,4] = 0
    # A[4,0] = 0
    # np.fill_diagonal(A, 0)
    #
    # Y = np.array([[0,0,0,1,1]]).reshape((5,1))
    #
    # print(A)
    # print(Y)
    #
    # Encoder_case5 = Encoder_case(A,Y,5)
    #
    # from Main.Python.DataPreprocess import DataPreprocess
    #
    # Dataset = DataPreprocess(Encoder_case5, Laplacian = False, DiagA = False)
    # print(Dataset.X)
    # print(Dataset.Y)
    # print(Dataset.n)

    # print("Running graph_encoder_embed()")

    # Z, W = graph_encoder_embed(Dataset.X[0], Dataset.Y, Dataset.n, Correlation = False)
    # print(Z)
    # print(W)


    print("Loading custom input graph")

    G_edgelist = np.loadtxt("../../Data/facebook_combined.txt")

    # Add column of ones - weights
    G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))

    n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices

    Y = np.array([i % 10 for i in range(n)]).reshape((n, 1)) # Init Y to nonsense TODO change to random sampling

    Z, W = graph_encoder_embed(G_edgelist, Y, n, Correlation = False)
    print(Z)
    print(W)
