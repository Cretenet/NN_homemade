import numpy as np

from activation_functions import RELU, sigmoid, softmax, tanh
from neural_network import NeuralNetwork


def test_forward_propagation():
    NN = NeuralNetwork(2, "input layer")
    NN.add_hidden_layer("first hidden layer", 2, "RELU")
    NN.add_output_layer("output layer", 1, "sigmoid")
    NN.weights[1] = np.array([[0.1, 0.2], [0.3, 0.4]])
    NN.weights[2] = np.array([[0.5, 0.6]])
    V = NN.forward_propagation(np.array([[0.5, -0.1], [0.7, 0.8]]))
    assert np.allclose(np.array([[0.5873448815653372, 0.5619303515188666]]), V)


def test_forward_propagation_random():
    NN = NeuralNetwork(256, "input layer")
    NN.add_hidden_layer("first hidden layer", 128, "RELU")
    NN.add_hidden_layer("second hidden layer", 64, "tanh")
    NN.add_hidden_layer("third hidden layer", 32, "sigmoid")
    NN.add_output_layer("output layer", 5, "softmax")
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
