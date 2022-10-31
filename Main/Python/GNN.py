import copy

import numpy as np
from sklearn.metrics import adjusted_rand_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from Main.Python.Hyperperameters import Hyperperameters


class GNN:
    def __init__(self, DataSets, **kwargs):
        GNN.kwargs = self.kwargs_construct(**kwargs)
        GNN.DataSets = DataSets
        GNN.hyperM = Hyperperameters()
        GNN.model = self.GNN_model()  #model summary: GNN.model.summary()
        GNN.meanSS = 0  # initialize the self-defined critirion meanSS

    def kwargs_construct(self, **kwargs):
        defaultKwargs = {'Learner': 1,                    # GNN_Leaner
                         'LearnerIter': 0,                # GNN_complete, GNN_Iter
                         "Replicates": 3                  # GNN_Iter
                         }
        kwargs = { **defaultKwargs, **kwargs}  # update the args using input_args
        return kwargs


    def GNN_model(self):
        """
          build GNN model
        """
        hyperM = self.hyperM
        DataSets = self.DataSets

        z_train = DataSets.z_train
        k = DataSets.k

        feature_num = z_train.shape[1]

        model = keras.Sequential([
            keras.layers.Flatten(input_shape = (feature_num,)),  # input layer
            keras.layers.Dense(hyperM.hidden, activation='relu'),  # hidden layer -- no tansig activation function in Keras, use relu instead
            keras.layers.Dense(k, activation='softmax') # output layer, matlab used softmax for patternnet default ??? max(opts.neuron,K)? opts
        ])

        optimizer = keras.optimizers.Adam(learning_rate = hyperM.learning_rate)

        model.compile(optimizer='adam',
                      loss=hyperM.loss,
                      metrics=['accuracy'])

        return model


    def GNN_run(self, k, z_train, y_train_one_hot, z_unlabel):
        """
          Train and test directly.
          Do not learn from the unknown labels.
        """
        gnn = copy.deepcopy(self)
        hyperM = gnn.hyperM
        model = gnn.model

        history = model.fit(z_train, y_train_one_hot,
                            epochs=hyperM.epochs,
                            validation_split=hyperM.val_split,
                            verbose=0)

        train_acc = history.history['accuracy'][-1]

        predict_probs = None
        pred_class = None
        pred_class_prob = None
        if type(z_unlabel) == np.ndarray:
            # predict_probas include probabilities for all classes for each node
            predict_probs = model.predict(z_unlabel)
            # assign the classes with the highest probability
            pred_class = np.argmax(predict_probs, axis=1)
            # the corresponding probabilities of the predicted classes
            pred_class_prob = predict_probs[range(len(predict_probs)),pred_class]

        gnn.model = model
        gnn.train_acc = train_acc
        gnn.pred_probs = predict_probs
        gnn.pred_class = pred_class
        gnn.pred_class_prob = pred_class_prob


        return gnn

    def GNN_Direct(self, DataSets, y_train_one_hot):
        """
          This function can run:
          1. by itself, when interation is set to False (<1)
          2. inside GNN_Iter, when interation is set to True (>=1)

          Learner == 0: GNN, but not learn from the known label
          Learner == 2: GNN, and learn unknown labels
        """
        Learner = self.kwargs["Learner"]

        k = DataSets.k
        z_train = DataSets.z_train
        Y = DataSets.Y
        z_unlabel = DataSets.z_unlabel
        ind_unlabel = DataSets.ind_unlabel

        gnn = self.GNN_run(k, z_train, y_train_one_hot, z_unlabel)

        if Learner == 0:
            # do not learn unknown label.
            pass

        if Learner == 2:
            # learn unknown label based on the known label
            # replace the unknown label in Y with predicted labels
            pred_class = gnn.pred_class
            Y[ind_unlabel, 0] = pred_class

        gnn.Y = Y

        return gnn


    def GNN_Iter(self, DataSets):
        """
          Run this function when interation is set, which is >=1.

          1. randomly assign labels to the unknown labels, get Y_temp
          2. get Y_one_hot for the Y_temp
          3. get Z from graph_encod function with X and Y_temp
          within each loop:
            use meanSS as its criterion to decide if the update is needed
              update Y_one_hot for the unknown labels with predict probabilities of each classes
              update Y with the highest possible predicted labels
              update z_train and z_unlabel from graph encoder embedding using the updated Y
              train the model with the updated z_train and Y_one_hot
        """

        kwargs = self.kwargs
        meanSS = self.meanSS

        k = DataSets.k
        Y = DataSets.Y
        ind_unlabel = DataSets.ind_unlabel


        y_temp = np.copy(Y)
        DataSets.y_temp = y_temp


        for i in range(kwargs["Replicates"]):
            # assign random integers in [1,K] to unassigned labels
            r = [i for i in range(k)]

            ran_int = np.random.choice(r, size=(len(ind_unlabel),1))

            y_temp[ind_unlabel] = ran_int

            for j in range(kwargs["LearnerIter"]):
                if j ==0:
                    # first iteration need to split the y_temp for training etc.
                    # use reset to add z_train, z_unlabel, y_temp_one_hot, to the dataset
                    DataSets = DataSets.DataSets_reset("y_temp")
                    # Convert targets into one-hot encoded format
                    y_temp_one_hot = to_categorical(y_temp)
                    # initialize y_temp_one_hot in the first loop
                    DataSets.y_temp_one_hot = y_temp_one_hot
                if j > 0:
                    # update z_train, z_unlabel, and y_temp_train_one_hot to the dataset
                    DataSets = DataSets.DataSets_reset("y_temp")
                # all the gnn train on y_train_one_hot
                gnn = self.GNN_Direct(DataSets, DataSets.Y_train_one_hot)
                predict_probs = gnn.pred_probs
                pred_class = gnn.pred_class
                pred_class_prob = gnn.pred_class_prob

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
                    # assign the probabilites for each class to the temp y_one_hot
                    DataSets.y_temp_one_hot[ind_unlabel] = predict_probs
                    # assgin the predicted classes to the temp Y unknown labels
                    DataSets.y_temp[ind_unlabel, 0] = pred_class
                    # # assign the highest possibility of the class to Y_temp
                    # Y_temp[ind_unlabel, 0] = pred_class_prob
            minP = np.mean(pred_class_prob) - 3*np.std(pred_class_prob)
            if minP > meanSS:
                meanSS = minP
                Y = DataSets.y_temp

            gnn.Y = Y
            gnn.meanSS = meanSS
            return gnn


    def GNN_complete(self):
        """
          if LearnerIter set to False(<1):
            run GNN_Direct() with no iteration
          if LearnerIter set to True(>=1):
            run GNN_Iter(), which starts with radomly assigned k to unknown labels

        """
        kwargs = self.kwargs

        DataSets = self.DataSets
        y_train = DataSets.Y_train

        if kwargs["LearnerIter"] < 1:
            # Convert targets into one-hot encoded format
            y_train_one_hot = to_categorical(y_train)
            gnn = self.GNN_Direct(DataSets, y_train_one_hot)
        else:
            gnn = self.GNN_Iter(DataSets)

        return gnn
