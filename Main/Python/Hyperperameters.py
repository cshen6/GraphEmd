class Hyperperameters:
    """
      define perameters for GNN.
      default values are for GNN learning -- "Leaner" ==2:
        embedding via partial label, then learn unknown label via two-layer NN

    """
    def __init__(self):
        # there is no scaled conjugate gradiant in keras optimiser, use defualt instead
        # use whatever default
        self.learning_rate = 0.01  # Initial learning rate.
        self.epochs = 100 #Number of epochs to train.
        self.hidden = 20 #Number of units in hidden layer
        self.val_split = 0.1 #Split 10% of training data for validation
        self.loss = 'categorical_crossentropy' # loss function
