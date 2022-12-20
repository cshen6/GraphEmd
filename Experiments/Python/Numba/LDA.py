import copy

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score


class LDA:
    def __init__(self, DataSets, **kwargs):
        LDA.kwargs = self.kwargs_construct(**kwargs)
        LDA.DataSets = DataSets
        LDA.model = LinearDiscriminantAnalysis()  # asssume spseudolinear is its default setting
        LDA.meanSS = 0  # initialize the self-defined critirion meanSS

    def kwargs_construct(self, **kwargs):
        defaultKwargs = {'Learner': 1,                            # LDA_Leaner
                         'LearnerIter': 0, "Replicates": 3     # LDA_Iter
                         }
        kwargs = { **defaultKwargs, **kwargs}  # update the args using input_args
        return kwargs

    def LDA_Learner(self, DataSets):
        """
          run this function when Learner set to 1.
          embedding via partial label, then learn unknown label via LDA.
        """
        lda = copy.deepcopy(self)
        z_train = DataSets.z_train
        y_train = DataSets.Y_train
        ind_unlabel = DataSets.ind_unlabel
        z_unlabel = DataSets.z_unlabel
        Y = DataSets.Y

        model = self.model
        model.fit(z_train,y_train)
        # train_acc = model.score(z_train,y_train)

        # for semi-supervised learning
        if type(z_unlabel) == np.ndarray:
            # predict_probas include probabilities for all classes for each node
            pred_probs = model.predict_proba(z_unlabel)
            # assign the classes with the highest probability
            pred_class = model.predict(z_unlabel)
            # the corresponding probabilities of the predicted classes
            pred_class_prob = pred_probs[range(len(pred_probs)),pred_class]
            # assign the predicted class to Y
            Y[ind_unlabel, 0] = pred_class
            lda.Y = Y
            lda.pred_class = pred_class
            lda.pred_class_prob = pred_class_prob

        lda.model = model
        return lda

    def LDA_Iter(self):
        """
          run this function when Learner set to 1, and LeanerIter is True(>=1)
          ramdonly assign labels to the unknownlabel.
          embedding via partial label, then learn unknown label via LDA.
        """

        kwargs = self.kwargs
        meanSS = self.meanSS
        DataSets = self.DataSets

        k = DataSets.k
        Y = DataSets.Y
        ind_unlabel = DataSets.ind_unlabel

        y_temp = np.copy(Y)

        for i in range(kwargs["Replicates"]):
            # assign random integers in [1,K] to unassigned labels
            r = [i for i in range(k)]

            ran_int = np.random.choice(r, size=(len(ind_unlabel),1))

            y_temp[ind_unlabel] = ran_int

            DataSets.y_temp = y_temp

            for j in range(kwargs["LearnerIter"]):
                # use reset to add z_train, z_unlabel, to the dataset
                DataSets = DataSets.DataSets_reset("y_temp")
                # all train on y_train
                lda = self.LDA_Learner(DataSets)
                pred_class = lda.pred_class
                pred_class_prob = lda.pred_class_prob

                # z_unknown is initialized with none, so the pred_class may be none
                # This will not happen for the semi version,
                # since the unknown size should not be none for the semi version
                if type(pred_class) == np.ndarray:
                    # if there are unkown labels and predicted labels are available
                    # check if predicted_class are the same as the random integers
                    # if so, stop the iteration in "LearnerIter" loop
                    # shape (n,) is required for adjusted_rand_score()
                    if adjusted_rand_score(ran_int.reshape((-1,)), pred_class) == 1:
                        break
                    # assgin the predicted classes to the temp Y unknown labels
                    DataSets.y_temp[ind_unlabel, 0] = pred_class
                    # # assign the highest possibility of the class to Y_temp
                    # Y_temp[ind_unlabel, 0] = pred_class_prob
            minP = np.mean(pred_class_prob) - 3*np.std(pred_class_prob)
            if minP > meanSS:
                meanSS = minP
                Y = DataSets.y_temp

            lda.Y = Y
            lda.meanSS = meanSS
            return lda


        ############-----------------LDA_end-----------------------------###############
