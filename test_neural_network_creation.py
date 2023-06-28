import numpy as np
from pytest import raises

from neural_network import NeuralNetwork


def test_init_positive_integer_input():
    NN = NeuralNetwork(5)
    assert NN.nbLayers == 0, "Incorrect number of layers"
    assert NN.layers_size == [5], "Incorrect layers size"
    assert NN.names == ["input_layer"], "Incorrect layer name"
    assert NN.activation_functions == [
        "identity"
    ], "Incorrect activation functions"
    assert not NN.output_layer_exists, "Output layer exists"
    assert NN.weights == {}, "Weights not empty"
    assert NN.biases == {}, "Biases not empty"
    assert NN.activation == {}, "Activations not empty"
    assert NN.Z == {}, "Z not empty"
    assert NN.db == {}, "db not empty"
    assert NN.dW == {}, "dW not empty"
    print("test_init_positive_integer_input passed")


def test_init_custom_name():
    NN = NeuralNetwork(5, "custom_layer")
    assert NN.names == ["custom_layer"], "Incorrect layer name"
    print("test_init_custom_name passed")


def test_init_bad_input():
    bad_integers = [-1, 0]
    for value in bad_integers:
        with raises(ValueError):
            NeuralNetwork(value)
    non_integers = [5.5, "5", [5], None]
    for value in non_integers:
        with raises(TypeError):
            NeuralNetwork(value)


def test_add_hidden_layer():
    NN = NeuralNetwork(5)
    NN.add_hidden_layer(4, "RELU", "first_hidden_layer")
    assert NN.nbLayers == 1
    assert NN.layers_size == [5, 4]
    assert NN.names == ["input_layer", "first_hidden_layer"]
    assert NN.activation_functions == ["identity", "RELU"]


def test_add_hidden_layer_zero_neurons():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_hidden_layer(0, "RELU", "hidden_layer")


def test_add_hidden_layer_negative_neurons():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_hidden_layer(-1, "RELU", "hidden_layer")


def test_add_hidden_layer_non_integer_neurons():
    NN = NeuralNetwork(5)
    with raises(TypeError):
        NN.add_hidden_layer(5.5, "RELU", "hidden_layer")
    with raises(TypeError):
        NN.add_hidden_layer("5", "RELU", "hidden_layer")
    with raises(TypeError):
        NN.add_hidden_layer([5], "RELU", "hidden_layer")
    with raises(TypeError):
        NN.add_hidden_layer(None, "RELU", "hidden_layer")


def test_add_hidden_layer_unsupported_activation():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_hidden_layer(5, "unsupported_activation", "hidden_layer")


def test_add_hidden_layer_after_output_layer_exists():
    NN = NeuralNetwork(5)
    NN.output_layer_exists = True  # Simulating an output layer
    with raises(ValueError):
        NN.add_hidden_layer(5, "RELU", "hidden_layer")


def test_add_output_layer():
    NN = NeuralNetwork(5)
    NN.add_output_layer(2, "RELU", "MSE", "output")
    assert NN.output_layer_exists
    assert NN.cost_function == "MSE"
    assert NN.nbLayers == 1
    assert NN.layers_size == [5, 2]
    assert NN.names == ["input_layer", "output"]
    assert NN.activation_functions == ["identity", "RELU"]


def test_add_output_layer_bad_nb_outputs():
    NN = NeuralNetwork(5)
    badvalues = [-1, 0]
    for badvalue in badvalues:
        with raises(ValueError):
            NN.add_output_layer(badvalue, "RELU", "MSE", "output")
    badtypes = [1.5, "2", [2], None]
    for badtype in badtypes:
        with raises(TypeError):
            NN.add_output_layer(badtype, "RELU", "MSE", "output")


def test_add_output_layer_unsupported_activation():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_output_layer(2, "unsupported_activation", "MSE", "output")


def test_add_output_layer_unsupported_cost():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_output_layer(2, "RELU", "unsupported_cost", "output")


def test_add_output_layer_softmax_non_cross_entropy():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_output_layer(2, "softmax", "MSE", "output")


def test_add_output_layer_binary_cross_entropy_more_than_one_output():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_output_layer(2, "sigmoid", "binary_cross_entropy", "output")


def test_add_output_layer_cross_entropy_non_softmax_sigmoid():
    NN = NeuralNetwork(5)
    with raises(ValueError):
        NN.add_output_layer(2, "RELU", "cross_entropy", "output")


def test_add_output_layer_after_output_layer_exists():
    NN = NeuralNetwork(5)
    NN.output_layer_exists = True
    with raises(ValueError):
        NN.add_output_layer(2, "RELU", "MSE", "output")


def test_initialize_weights_relu():
    NN = NeuralNetwork(5)
    NN.add_hidden_layer(4, "RELU", "hidden_layer_1")
    assert NN.weights[1].shape == (4, 5)
    assert np.all(NN.biases[1] == np.zeros((4, 1)))


def test_initialize_weights_non_relu():
    NN = NeuralNetwork(5)
    NN.add_hidden_layer(4, "sigmoid", "hidden_layer_1")
    assert NN.weights[1].shape == (4, 5)
    assert np.all(NN.biases[1] == np.zeros((4, 1)))
