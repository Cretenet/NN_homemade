import numpy as np

import activation_functions as af
import cost_functions as cf


class NeuralNetwork:
    """
    A class used to create an artificial neural network.

    Parameters
    ----------
    input_size : int
        Number of neurons in the input layer.
    input_layer_name : str, optional
        Name of the input layer. Default is "input_layer".

    Attributes
    ----------
    nbLayers : int
        Number of layers in the neural network.
    layers_size : list
        List of sizes of each layer.
    input_size : int
        Size of the input layer.
    names : list
        Names of layers.
    activation_functions : list
        Activation functions used for each layer.
    output_layer_exists : bool
        Checks if output layer exists.
    weights : dict
        Weights for each layer.
    biases : dict
        Biases for each layer.
    activation : dict
        Activations for each layer.
    Z : dict
        Pre-activation values for each layer.
    derivative_activation : dict
        Derivatives of activation function for each layer.
    db : dict
        Gradients of loss function with respect to biases.
    dW : dict
        Gradients of loss function with respect to weights.
    """

    def __init__(self, input_size: int, input_layer_name="input_layer"):
        if not isinstance(input_size, int):
            raise TypeError("input_size must be an integer")
        if input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        self.nbLayers = 0  # By convention, the input layer is not counted
        self.layers_size = [input_size]
        self.input_size = input_size
        self.names = [input_layer_name]
        self.activation_functions = ["identity"]
        self.output_layer_exists = False
        self.weights = {}
        self.biases = {}
        self.activation = {}
        self.Z = {}
        self.derivative_activation = {}
        self.db = {}  # Gradient of the loss function w.r.t the biases
        self.dW = {}  # Gradient of the loss function w.r.t the weights

    def add_hidden_layer(
        self, nb_neurons: int, activation: str, name="hidden_layer"
    ):
        """
        Adds a hidden layer to the neural network.

        Parameters
        ----------
        nb_neurons : int
            Number of neurons in the hidden layer.
        activation : str
            Activation function to be used in the hidden layer.
        name : str, optional
            Name of the hidden layer. Default is "hidden_layer".
        """
        if not isinstance(nb_neurons, int):
            raise TypeError("nb_neurons must be an integer")
        if nb_neurons <= 0:
            raise ValueError("nb_neurons must be a strictly positive integer")
        if activation not in ["RELU", "sigmoid", "tanh", "identity"]:
            raise ValueError(
                "The chosen activation function is not supported."
                + "Please choose between RELU, sigmoid, tanh, identity."
            )
        if not self.output_layer_exists:
            self.nbLayers += 1
            self.layers_size.append(nb_neurons)
            self.names.append(name)
            self.activation_functions.append(activation)
            self.initialize_weights(
                self.activation_functions[-1],
                self.nbLayers,
                self.layers_size[self.nbLayers - 1],
                nb_neurons,
            )
        else:
            raise ValueError(
                "The output layer has already been created."
                + "You cannot add a hidden layer anymore."
            )

    def add_output_layer(
        self, nb_outputs: int, activation: str, cost: str, name="output_layer"
    ):
        """
        Adds an output layer to the neural network.

        Parameters
        ----------
        nb_outputs : int
            Number of outputs in the output layer.
        activation : str
            Activation function to be used in the output layer.
        cost : str
            Cost function to be used for training.
        name : str, optional
            Name of the output layer. Default is "output_layer".
        """
        if not isinstance(nb_outputs, int):
            raise TypeError("nb_outputs must be an integer")
        if nb_outputs <= 0:
            raise ValueError("nb_outputs must be a strictly positive integer")
        if activation not in [
            "RELU",
            "sigmoid",
            "tanh",
            "softmax",
            "identity",
        ]:
            raise ValueError(
                "The chosen activation function is not supported."
                + "Please choose between RELU, sigmoid, "
                + "tanh, softmax, identity."
            )
        if cost not in ["MSE", "cross_entropy", "binary_cross_entropy"]:
            raise ValueError(
                "The chosen cost function is not supported."
                + "Please choose between MSE, cross_entropy, "
                + "binary_cross_entropy."
            )
        if activation == "softmax" and cost != "cross_entropy":
            raise ValueError(
                "The softmax activation function is only compatible with"
                + " the cross entropy cost function."
            )
        if nb_outputs != 1 and cost == "binary_cross_entropy":
            raise ValueError(
                "The binary cross entropy cost function is only compatible"
                + " with one output."
            )
        if cost in [
            "cross_entropy",
            "binary_cross_entropy",
        ] and activation not in ["softmax", "sigmoid"]:
            raise ValueError(
                "The cross entropy and binary cross entropy cost functions"
                + " are only compatible with the softmax and sigmoid"
                + " activation functions."
            )
        if not self.output_layer_exists:
            self.cost_function = cost
            self.output_layer_exists = True
            self.nbLayers += 1
            self.layers_size.append(nb_outputs)
            self.names.append(name)
            self.activation_functions.append(activation)
            self.initialize_weights(
                self.activation_functions[-1],
                self.nbLayers,
                self.layers_size[self.nbLayers - 1],
                nb_outputs,
            )
        else:
            raise ValueError(
                "The output layer has already been created."
                + "You cannot add another output layer."
            )

    def initialize_weights(
        self,
        activation_function: str,
        index: int,
        nb_first_layer: int,
        nb_second_layer: int,
    ):
        """
        Initializes weights and biases for a specific layer.

        Parameters
        ----------
        activation_function : str
            Activation function to be used in the layer.
        index : int
            Index of the layer.
        nb_first_layer : int
            Number of neurons in the first layer.
        nb_second_layer : int
            Number of neurons in the second layer.
        """
        m = 1
        if activation_function == "RELU":
            m = 2
        self.weights[index] = np.random.normal(
            0, m / self.layers_size[0], (nb_second_layer, nb_first_layer)
        )
        self.biases[index] = np.zeros((nb_second_layer, 1))

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
    ):
        """
        Trains the neural network on a given dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Input of the neural network.
        Y : numpy.ndarray
            Output of the neural network.
        epochs : int
            Number of epochs for training.
        lr : float
            Learning rate for training.
        batch_size : int
            Size of the batches for training.
        """
        # We first verify that the input has the right format
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays")
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-dimensional")
        if X.shape[0] != self.input_size:
            raise ValueError("X must have as much rows as the input layer")
        if Y.shape[0] != self.layers_size[-1]:
            raise ValueError("Y must have as much rows as the output layer")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of columns")
        if epochs <= 0:
            raise ValueError("epochs must be a strictly positive integer")
        if lr <= 0:
            raise ValueError("lr must be a strictly positive float")
        if batch_size <= 0:
            raise ValueError("batch_size must be a strictly positive integer")
        if batch_size > X.shape[1]:
            raise ValueError(
                "batch_size must be lower than or equal to "
                "the number of samples"
            )
        # We then start the training
        for epoch in range(epochs):
            # We shuffle the dataset
            X, Y = self.shuffle_dataset(X, Y)
            # We divide the dataset into batches
            X_batches, Y_batches = self.divide_batches(X, Y, batch_size)
            # We train the neural network on each batch
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                self.train_batch(X_batch, Y_batch, lr)
            print(
                f"Epoch {epoch + 1} / {epochs} completed, "
                f"loss: {self.evaluate(Y=Y, Yhat=self.forward_propagation(X))}"
                f" accuracy: "
                f"{self.accuracy(Y=Y, Yhat=self.forward_propagation(X))}"
            )

    def shuffle_dataset(self, X: np.ndarray, Y: np.ndarray):
        """
        Shuffles the dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Input of the neural network.
        Y : numpy.ndarray
            Output of the neural network.

        Returns
        -------
        numpy.ndarray
            Shuffled input.
        numpy.ndarray
            Shuffled output.
        """
        permutation = np.random.permutation(X.shape[1])
        return X[:, permutation], Y[:, permutation]

    def divide_batches(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
        """
        Divides the dataset into batches.

        Parameters
        ----------
        X : numpy.ndarray
            Input of the neural network.
        Y : numpy.ndarray
            Output of the neural network.
        batch_size : int
            Size of the batches.

        Returns
        -------
        numpy.ndarray
            Input batches.
        numpy.ndarray
            Output batches.
        """
        nb_batches = X.shape[1] // batch_size
        X_batches = np.array_split(X, nb_batches, axis=1)
        Y_batches = np.array_split(Y, nb_batches, axis=1)
        return X_batches, Y_batches

    def train_batch(self, X: np.ndarray, Y: np.ndarray, lr: float):
        """
        Trains the neural network on a given batch.

        Parameters
        ----------
        X : numpy.ndarray
            Input of the neural network.
        Y : numpy.ndarray
            Output of the neural network.
        lr : float
            Learning rate for training.
        """
        # We first verify that the input has the right format
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays")
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-dimensional")
        if X.shape[0] != self.input_size:
            raise ValueError("X must have as much rows as the input layer")
        if Y.shape[0] != self.layers_size[-1]:
            raise ValueError("Y must have as much rows as the output layer")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of columns")
        if lr <= 0:
            raise ValueError("lr must be a strictly positive float")
        # We then start the training
        self.forward_propagation(X)
        self.backward_propagation(Y)
        self.update_parameters(lr)

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the neural network.

        Parameters
        ----------
        X : numpy.ndarray
            Input to the neural network.

        Returns
        -------
        numpy.ndarray
            Output from the neural network.
        """
        # We first verify that the input has the right format
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] != self.input_size:
            raise ValueError("X must have as much rows as the input layer")
        # Start the propagation
        first_layer = X
        self.activation[0] = first_layer
        for layer in range(1, self.nbLayers + 1):
            W = self.weights[layer]
            b = self.biases[layer]
            Z = W @ first_layer + b
            second_layer = af.activation_function(
                Z, self.activation_functions[layer]
            )
            # The following line is useful for backward propagation
            self.activation[layer] = second_layer
            self.Z[layer] = Z
            first_layer = second_layer  # Restart with the next one
        return second_layer

    def backward_propagation(self, Y: np.ndarray):
        """
        Performs backward propagation through the neural network.

        Parameters
        ----------
        Y : numpy.ndarray
            Ground truth values for the inputs.
        """

        # We first verify that Y has the right format
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if Y.ndim != 2:
            raise ValueError("Y must be 2-dimensional")
        if Y.shape != self.activation[self.nbLayers].shape:
            raise ValueError("Y must have the shape (nb_units, N)")
        # Initialize
        Yhat = self.activation[self.nbLayers]
        if (
            self.activation_functions[-1] == "softmax"
            and self.cost_function == "cross_entropy"
        ) or (
            self.activation_functions[-1] == "sigmoid"
            and self.cost_function == "binary_cross_entropy"
        ):
            dZ = Yhat - Y
        else:
            dZ = af.derivative_activation_function(
                Z=self.Z[self.nbLayers],
                name=self.activation_functions[self.nbLayers],
            ) * cf.derivative_cost_function(Y, Yhat, self.cost_function)
        # Backward propagation
        for layer in range(self.nbLayers, 0, -1):
            m = dZ.shape[1]
            self.db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            self.dW[layer] = (1 / m) * dZ @ self.activation[layer - 1].T
            if layer > 1:
                dZ = (
                    self.weights[layer].T
                    @ dZ
                    * af.derivative_activation_function(
                        Z=self.Z[layer - 1],
                        name=self.activation_functions[layer - 1],
                    )
                )

    def update_parameters(self, lr: float):
        """
        Updates the parameters of the neural network.

        Parameters
        ----------
        lr : float
            Learning rate for training.
        """
        # We first verify that lr has the right format
        if not isinstance(lr, float):
            raise TypeError("lr must be a float")
        if lr <= 0:
            raise ValueError("lr must be a strictly positive float")
        # Update the parameters
        for layer in range(1, self.nbLayers + 1):
            self.weights[layer] -= lr * self.dW[layer]
            self.biases[layer] -= lr * self.db[layer]

    def evaluate(self, Y: np.ndarray, Yhat: np.ndarray) -> float:
        """
        Evaluates the performance of the neural network.

        Parameters
        ----------
        Y : numpy.ndarray
            Ground truth values for the inputs.
        Yhat : numpy.ndarray
            Predicted values for the inputs.

        Returns
        -------
        float
            Cost of the current state of the neural network.
        """
        # We first verify that Y and Yhat have the right format
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if not isinstance(Yhat, np.ndarray):
            raise TypeError("Yhat must be a numpy array")
        if Y.ndim != 2:
            raise ValueError("Y must be 2-dimensional")
        if Yhat.ndim != 2:
            raise ValueError("Yhat must be 2-dimensional")
        if Y.shape != Yhat.shape:
            raise ValueError("Y and Yhat must have the same shape")
        # Compute the cost
        return cf.cost_function(Y, Yhat, self.cost_function)

    def accuracy(self, Y, Yhat):
        """
        Computes the accuracy of the neural network.

        Parameters
        ----------
        Y : numpy.ndarray
            Ground truth values for the inputs.
        Yhat : numpy.ndarray
            Predicted values for the inputs.

        Returns
        -------
        float
            Accuracy of the current state of the neural network.
        """
        # We first verify that Y and Yhat have the right format
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if not isinstance(Yhat, np.ndarray):
            raise TypeError("Yhat must be a numpy array")
        if Y.ndim != 2:
            raise ValueError("Y must be 2-dimensional")
        if Yhat.ndim != 2:
            raise ValueError("Yhat must be 2-dimensional")
        if Y.shape != Yhat.shape:
            raise ValueError("Y and Yhat must have the same shape")
        # Compute the accuracy
        return np.mean(np.argmax(Y, axis=0) == np.argmax(Yhat, axis=0))
