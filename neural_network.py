import numpy as np

import activation_functions as af


class NeuralNetwork:
    """
    The neural_network class allows the user to create, train, and use an
    artificial neural network. The layers must be created in the right order,
    the input layer is automatically created. The user can create n hidden
    layers by calling n times add_hidden_layer(). The user must create an
    output layer as last layer with add_output_layer().
    """

    def __init__(self, input_size, input_layer_name="input_layer"):
        self.nbLayers = 0  # By convention, the input layer is not counted
        self.layers_size = [input_size]
        self.input_size = input_size
        self.names = [input_layer_name]
        self.activation_functions = ["identity"]
        self.output_layer_exists = False
        self.weights = {}
        self.biases = {}
        self.activation = {}
        self.derivative_activation = {}
        self.dl_db = {}  # Gradient of the loss function w.r.t the biases
        self.dl_dw = {}  # Gradient of the loss function w.r.t the weights

    def add_hidden_layer(self, name, nb_neurons=1, activation_function="RELU"):
        if not self.output_layer_exists:
            self.nbLayers += 1
            self.layers_size.append(nb_neurons)
            self.names.append(name)
            if activation_function in ["RELU", "sigmoid", "tanh", "softmax"]:
                self.activation_functions.append(activation_function)
            else:
                self.activation_functions.append("RELU")
                print(
                    "The chosen activation function is not supported, "
                    + "RELU is used instead."
                )
            self.initialize_weights(
                self.activation_functions[-1],
                self.nbLayers,
                self.layers_size[self.nbLayers - 1],
                nb_neurons,
            )
        else:
            print(
                "It is not possible to add a hidden layer "
                + "after the output layer. "
                + "The layers must be added in the right order."
            )

    def add_output_layer(
        self, name="output_layer", nb_outputs=1, activation_function="sigmoid"
    ):
        self.output_layer_exists = True
        self.nbLayers += 1
        self.layers_size.append(nb_outputs)
        self.names.append(name)
        if activation_function in ["RELU", "sigmoid", "tanh", "softmax"]:
            self.activation_functions.append(activation_function)
        else:
            self.activation_functions.append("softmax")
            print(
                "The chosen activation function is not supported,"
                + " sigmoid is used instead."
            )
        self.initialize_weights(
            self.activation_functions[-1],
            self.nbLayers,
            self.layers_size[self.nbLayers - 1],
            nb_outputs,
        )

    def initialize_weights(
        self, activation_function, index, nb_first_layer, nb_second_layer
    ):
        if activation_function == "RELU":
            m = 2
        elif activation_function in ["sigmoid", "tanh", "softmax"]:
            m = 1
        self.weights[index] = np.random.normal(
            0, m / self.layers_size[0], (nb_second_layer, nb_first_layer)
        )
        self.biases[index] = np.zeros((nb_second_layer, 1))

    def forward_propagation(self, input):
        # We first verify that the input has the right format
        if not isinstance(input, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if input.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")
        # Start the propagation
        first_layer = input
        self.activation[0] = first_layer
        for layer in range(1, self.nbLayers + 1):
            W = self.weights[layer]
            b = self.biases[layer]
            z = W @ first_layer + b
            second_layer = af.activation_function(
                z, self.activation_functions[layer]
            )
            # The following line is useful for backward propagation
            self.activation[layer] = second_layer
            first_layer = second_layer  # Restart with the next one
        return second_layer

    def evaluation(self, prediction, truth):
        prediction = np.reshape(prediction, (len(prediction), 1))
        truth = np.reshape(truth, (len(truth), 1))
        return np.sum(np.square(prediction - truth))  # Sum of squares

    def backward_propagation(self, truth):
        truth = np.reshape(truth, (len(truth), 1))
        index_left_layer = self.nbLayers - 2
        index_right_layer = self.nbLayers - 1
        left_layer_size = self.layers_size[index_left_layer]
        right_layer_size = self.layers_size[index_right_layer]
        self.dl_dw[self.nbLayers - 2] = np.zeros(
            (right_layer_size, left_layer_size)
        )
        self.dl_dw[self.nbLayers - 2][0][0] = (
            2
            * (self.activation[self.nbLayers - 1][0] - truth[0])
            * self.derivative_activation[self.nbLayers - 2][0]
            * self.activation[self.nbLayers - 2][0]
        )
        self.dl_dw[self.nbLayers - 2][1][0] = (
            2
            * (self.activation[self.nbLayers - 1][1] - truth[1])
            * self.derivative_activation[self.nbLayers - 2][1]
            * self.activation[self.nbLayers - 2][0]
        )
        previous_dl_da = 2 * (self.activation[self.nbLayers - 1] - truth)
        for layer in range(0, self.nbLayers - 1):  # Loop over the layers
            index_left_layer = self.nbLayers - 2 - layer
            index_right_layer = self.nbLayers - 1 - layer
            left_layer_size = self.layers_size[index_left_layer]
            right_layer_size = self.layers_size[index_right_layer]
            self.dl_dw[self.nbLayers - 2 - layer] = np.zeros(
                (right_layer_size, left_layer_size)
            )
            self.dl_db[self.nbLayers - 2 - layer] = np.zeros(
                (right_layer_size, 1)
            )
            for j in range(0, right_layer_size):
                self.dl_db[self.nbLayers - 2 - layer][j][0] = (
                    previous_dl_da[j]
                    * self.derivative_activation[self.nbLayers - 2 - layer][j]
                )
                for k in range(0, left_layer_size):
                    self.dl_dw[self.nbLayers - 2 - layer][j][k] = (
                        previous_dl_da[j]
                        * self.derivative_activation[
                            self.nbLayers - 2 - layer
                        ][j]
                        * self.activation[self.nbLayers - 2 - layer][k]
                    )
            dl_da = np.zeros((left_layer_size, 1))
            for k in range(0, left_layer_size):
                for j in range(0, right_layer_size):
                    dl_da[k][0] += (
                        self.weights[self.nbLayers - 2 - layer][j][k]
                        * self.derivative_activation[
                            self.nbLayers - 2 - layer
                        ][j]
                        * previous_dl_da[j]
                    )
            previous_dl_da = dl_da

    def info(self):
        message = (
            "The fully connected artificial neural network is composed of :\n"
        )
        message += (
            '\t - An input layer named "'
            + str(self.names[0])
            + '" expecting '
            + str(self.layers_size[0])
            + " inputs."
        )
        if self.nbLayers > 1:
            if self.output_layer_exists:
                for i in range(1, self.nbLayers - 1):
                    message += (
                        '\n\t - A hidden layer named "'
                        + str(self.names[i])
                        + '", containing '
                        + str(self.layers_size[i])
                        + " neurons, and using the "
                        + str(self.activation_functions[i - 1])
                        + " activation function."
                    )
                message += (
                    '\n\t - An output layer named "'
                    + str(self.names[-1])
                    + '", containing '
                    + str(self.layers_size[-1])
                    + " neurons, and using the "
                    + str(self.activation_functions[-1])
                    + " activation function."
                )
            else:
                for i in range(1, self.nbLayers):
                    message += (
                        '\n\t - A hidden layer named "'
                        + str(self.names[i])
                        + '", containing '
                        + str(self.layers_size[i])
                        + " neurons, and using the "
                        + str(self.activation_functions[i - 1])
                        + " activation function."
                    )
        message += "\n Now let us look at the different weight matrices :\n"
        print(message)
        for i in range(1, len(self.names)):
            print("W" + str(i) + " = ")
            print(self.weights[i - 1])
            print("\n")
