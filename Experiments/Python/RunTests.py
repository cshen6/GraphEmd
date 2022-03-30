import numpy as np


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


    from Main.Python.DataPreprocess import graph_encoder_embed

    Z, W = graph_encoder_embed(Dataset.X[0], Dataset.Y, Dataset.n, Correlation = False)
    print(Z)
    print(W)
