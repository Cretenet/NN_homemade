import numpy as np
from pytest import raises

from activation_functions import RELU, sigmoid, softmax, tanh
from neural_network import NeuralNetwork


def test_bad_input():
    NN = NeuralNetwork(input_size=2)
    NN.add_hidden_layer(nb_neurons=2, activation="RELU")
    NN.add_output_layer(cost="MSE", nb_outputs=1, activation="sigmoid")
    with raises(TypeError, match="X must be a numpy array"):
        NN.forward_propagation(X="not a numpy array")
    with raises(TypeError, match="X must be a numpy array"):
        NN.forward_propagation(X=[[0.5, -0.1], [0.7, 0.8]])
    with raises(ValueError, match="X must be 2-dimensional"):
        NN.forward_propagation(X=np.array([1, 2, 3]))
    with raises(
        ValueError,
        match="X must have as much rows as the input layer",
    ):
        NN.forward_propagation(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_fixed_values():
    NN = NeuralNetwork(input_size=2)
    NN.add_hidden_layer(nb_neurons=2, activation="RELU")
    NN.add_output_layer(cost="MSE", nb_outputs=1, activation="sigmoid")
    NN.weights[1] = np.array([[0.1, 0.2], [0.3, 0.4]])
    NN.weights[2] = np.array([[0.5, 0.6]])
    V = NN.forward_propagation(np.array([[0.5, -0.1], [0.7, 0.8]]))
    assert np.allclose(np.array([[0.5873448815653372, 0.5619303515188666]]), V)


def test_large_random_network():
    NN = NeuralNetwork(input_size=256)
    NN.add_hidden_layer(nb_neurons=128, activation="RELU")
    NN.add_hidden_layer(nb_neurons=64, activation="tanh")
    NN.add_hidden_layer(nb_neurons=32, activation="sigmoid")
    NN.add_output_layer(cost="MSE", nb_outputs=5, activation="softmax")
    W1 = NN.weights[1]
    assert W1.shape == (128, 256)
    W2 = NN.weights[2]
    assert W2.shape == (64, 128)
    W3 = NN.weights[3]
    assert W3.shape == (32, 64)
    W4 = NN.weights[4]
    assert W4.shape == (5, 32)
    b1 = NN.biases[1]
    assert b1.shape == (128, 1)
    b2 = NN.biases[2]
    assert b2.shape == (64, 1)
    b3 = NN.biases[3]
    assert b3.shape == (32, 1)
    b4 = NN.biases[4]
    assert b4.shape == (5, 1)
    X = np.random.rand(256, 3)
    V = NN.forward_propagation(X)
    assert V.shape == (5, 3)
    A1 = RELU(W1 @ X + b1)
    A2 = tanh(W2 @ A1 + b2)
    A3 = sigmoid(W3 @ A2 + b3)
    A4 = softmax(W4 @ A3 + b4)
    assert np.allclose(A4, V)
