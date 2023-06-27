import numpy as np
from pytest import raises

from neural_network import NeuralNetwork


def test_bad_input():
    NN = NeuralNetwork(input_size=2)
    NN.add_hidden_layer(nb_neurons=2, activation="RELU")
    NN.add_output_layer(cost="MSE", nb_outputs=1, activation="sigmoid")
    NN.forward_propagation(X=np.array([[0.5, -0.1], [0.7, 0.8]]))
    with raises(TypeError, match="Y must be a numpy array"):
        NN.backward_propagation(Y="not a numpy array")
    with raises(TypeError, match="Y must be a numpy array"):
        NN.backward_propagation(Y=[[0.5, -0.1], [0.7, 0.8]])
    with raises(ValueError, match="Y must be 2-dimensional"):
        NN.backward_propagation(Y=np.array([1, 2, 3]))
    with raises(
        ValueError,
    ):
        NN.backward_propagation(Y=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_binary_classification():
    hidden_activations = ["RELU", "tanh", "sigmoid", "identity"]
    costs = ["binary_cross_entropy"]
    output_activations = ["sigmoid"]
    nb_outputs = 1
    gradient_checking(
        nb_outputs, hidden_activations, output_activations, costs
    )


def test_multiclass_classification():
    hidden_activations = ["RELU", "tanh", "sigmoid", "identity"]
    costs = ["cross_entropy"]
    output_activations = ["softmax", "sigmoid"]
    nb_outputs = 3
    gradient_checking(
        nb_outputs, hidden_activations, output_activations, costs
    )


def test_regression():
    hidden_activations = ["RELU", "tanh", "sigmoid", "identity"]
    costs = ["MSE"]
    output_activations = ["identity", "RELU"]
    nb_outputs = 1
    gradient_checking(
        nb_outputs, hidden_activations, output_activations, costs
    )


def gradient_checking(
    nb_outputs, hidden_activations, output_activations, costs
):
    epsilon = 1e-7
    nb_samples = 10
    nb_inputs = 128
    for hidden_activation in hidden_activations:
        for output_activation in output_activations:
            for cost in costs:
                print(hidden_activation, output_activation, cost)
                NN = NeuralNetwork(
                    input_size=nb_inputs, input_layer_name="input"
                )
                NN.add_hidden_layer(
                    nb_neurons=64, activation=hidden_activation
                )
                NN.add_output_layer(
                    nb_outputs=nb_outputs,
                    activation=output_activation,
                    cost=cost,
                )
                X = np.random.rand(nb_inputs, nb_samples)
                # Y must be a one-hot vector (except if 1d output)
                Y = np.zeros((nb_outputs, nb_samples))
                for i in range(0, Y.shape[1]):
                    # One-hot vector (only one class is correct)
                    if Y.shape[0] > 1:
                        Y[np.random.randint(0, Y.shape[0])][i] = 1
                    else:
                        Y[0][i] = np.random.randint(0, 2)
                # Compute the gradient with NN
                NN.forward_propagation(X)
                NN.backward_propagation(Y)
                # Compute the gradient with numerical approximation
                for layer in range(1, NN.nbLayers + 1):
                    for i in range(0, NN.weights[layer].shape[0]):
                        # Weights ------------------------------------
                        for j in range(0, NN.weights[layer].shape[1]):
                            original_weight = NN.weights[layer][i][j]
                            # calculate the cost after adding epsilon
                            NN.weights[layer][i][j] = original_weight + epsilon
                            V2 = NN.forward_propagation(X)
                            cost_plus = NN.evaluate(Yhat=V2, Y=Y)
                            # calculate the cost after subtracting epsilon
                            NN.weights[layer][i][j] = original_weight - epsilon
                            V3 = NN.forward_propagation(X)
                            cost_minus = NN.evaluate(Yhat=V3, Y=Y)
                            # restore the original weight
                            NN.weights[layer][i][j] = original_weight
                            # compute the numerical gradient
                            derivative = (cost_plus - cost_minus) / (
                                2 * epsilon
                            )
                            print(derivative, NN.dW[layer][i][j])
                            assert np.isclose(derivative, NN.dW[layer][i][j])
                        # Biases ------------------------------------
                        original_bias = NN.biases[layer][i][0]
                        # calculate the cost after adding epsilon
                        NN.biases[layer][i][0] = original_bias + epsilon
                        V2 = NN.forward_propagation(X)
                        cost_plus = NN.evaluate(Yhat=V2, Y=Y)
                        # calculate the cost after subtracting epsilon
                        NN.biases[layer][i][0] = original_bias - epsilon
                        V3 = NN.forward_propagation(X)
                        cost_minus = NN.evaluate(Yhat=V3, Y=Y)
                        # restore the original weight
                        NN.biases[layer][i][0] = original_bias
                        # compute the numerical gradient
                        derivative = (cost_plus - cost_minus) / (2 * epsilon)
                        assert np.isclose(derivative, NN.db[layer][i][0])
