import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from Main.Python.Main import graph_encoder_embed, multi_graph_encoder_embed


class Clustering:
    """
      The input DataSets.X is the s*3 edg list
      The innput DataSets.Y can be:
      1. A given cluster size, e.g. [3], meaning in total 3 clusters
      2. A range of cluster sizes. e.g. [3-5], meaning there are possibly 3 to 5 clusters

    """
    def __init__(self, DataSets, **kwargs):
        self.kwargs = self.kwargs_construct(**kwargs)
        self.DataSets = DataSets
        self.cluster_size_range = self.cluster_size_check()
        self.K = DataSets.Y[0]


    def kwargs_construct(self, **kwargs):
        defaultKwargs = {'Correlation': True,'MaxIter': 50, 'MaxIterK': 5,'Replicates': 3}
        kwargs = { **defaultKwargs, **kwargs}
        return kwargs

    def cluster_size_check(self):
        DataSets = self.DataSets
        Y = DataSets.Y

        cluster_size_range = None # in case that Y is an empty array.

        if len(Y) == 1:
            cluster_size_range = False  # meaning the cluster size is known. e.g. [3]
        if len(Y) > 1:
            cluster_size_range = True   # meaning only know the rane of cluster size. e.g. [2, 3, 4, 5]

        return cluster_size_range

    def graph_encoder_cluster(self, K):
        """
          clustering function
        """
        DataSets = self.DataSets
        X = DataSets.X
        n = DataSets.n
        kwargs = self.kwargs


        minSS=-1
        Z = None
        W = None

        for i in range(kwargs['Replicates']):
            Y_temp = np.random.randint(K,size=(n,1))
            for r in range(kwargs['MaxIter']):
                [Zt,Wt] = multi_graph_encoder_embed(DataSets, Y_temp)

                if DataSets.attributes:
                    # add U to Z side by side
                    Zt = np.concatenate((Zt, DataSets.U), axis=1)
                kmeans = KMeans(n_clusters=K, max_iter = kwargs['MaxIter']).fit(Zt)
                labels = kmeans.labels_ # shape(n,)
                # sum_in_cluster = kmeans.inertia_ # sum of distance within cluster (k,1)
                dis_to_centors = kmeans.transform(Zt)
                # adjusted_rand_score() needs the shape (n,)
                if adjusted_rand_score(Y_temp.reshape(-1,), labels) == 1:
                    break
                else:
                    # we need labels to be the same shape as for Y(n,1) when assign
                    Y_temp = labels.reshape(-1,1)

                    # calculate score and compare with meanSS
            tmp = self.temp_score(dis_to_centors, K, labels, n)
            if (minSS == -1) or tmp < minSS:
                Z = Zt
                W = Wt
                minSS = tmp
                Y = labels
        return  Z, Y, W, minSS


    def temp_score(self, dis_to_centors, K, labels, n):
        """
          calculate:
          1. sum_in_cluster(1*k): the sum of the distance from the nodes to the centroid
          of its belonged cluster
          2. sum_in_cluster_norm(1*k): normalize the sum_in_cluster by the
          corresponding label count (how many nodes in each cluster)
          3. sum_not_in_cluster(1*k): the sum of the distance of the cluster
          centroid to the nodes that do not belong to the cluster
          4. sum_not_in_cluster_norm(1*k): normalize the sum_other_centroids by the
          counts of the nodes that do not belong to the cluster.
          5. temp score(1*k):
          (normalized sum in cluster / normalized sum not in cluster ) *
          (label count in cluster / total node number)
          6. get mean + 2 standard deviation of temp score, then return
        """
        label_count = np.bincount(labels)
        sum_in_cluster_squre = np.zeros((K,))

        dis_to_centors_squre = dis_to_centors**2

        for i in range(n):
            label = labels[i]
            sum_in_cluster_squre[label] += dis_to_centors_squre[i][label]

        # how to find out if the distance is squared, the current method doesn't do square root.
        sum_not_in_cluster = (np.sum(dis_to_centors_squre, axis=0) - sum_in_cluster_squre)**0.5

        sum_not_in_cluster_norm = sum_not_in_cluster/(n - label_count)
        sum_in_cluster_norm = sum_in_cluster_squre**0.5/label_count

        tmp = sum_in_cluster_norm / sum_not_in_cluster_norm * label_count / n
        tmp = np.mean(tmp) + 2*np.std(tmp)

        return tmp


    def cluster_main(self):
        K = self.K
        DataSets = self.DataSets
        X = DataSets.X
        n = DataSets.n

        kmax = np.amax(K)
        if n/kmax < 30:
            print('Too many clusters at maximum. Result may bias towards large K. Please make sure n/Kmax >30.')
        # when the cluster size is specified
        if not self.cluster_size_range:
            [Z,Y,W,meanSS]= self.graph_encoder_cluster(K[0])
        # when the range of cluster size is provided
        # columns are less than n/2 and kmax is less than max(n/2, 10)
        if self.cluster_size_range:
            k_range = len(K)
            if k_range < n/2 and kmax < max(n/2, 10):
                minSS = -1
                Z = 0
                W = 0
                meanSS = np.zeros((k_range,1))
                for i in range(k_range):
                    [Zt,Yt,Wt,tmp]= self.graph_encoder_cluster(K[i])
                    meanSS[i,0] = i
                    if (minSS == -1) or tmp < minSS:
                        minSS = tmp
                        Y = Yt
                        Z = Zt
                        W = Wt
        return Z, Y, W, meanSS
