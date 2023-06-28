import numpy as np
import pytest
from pytest import raises

from neural_network import NeuralNetwork


def test_shuffle_dataset():
    NN = NeuralNetwork(input_size=10)
    X = np.arange(30).reshape(3, 10)
    Y = np.arange(30).reshape(3, 10)
    X_shuffled, Y_shuffled = NN.shuffle_dataset(X, Y)
    assert np.all(X_shuffled == Y_shuffled)
    assert not np.all(X == X_shuffled)
    assert not np.all(Y == Y_shuffled)


def test_divide_batches():
    NN = NeuralNetwork(input_size=10)
    X = np.arange(30).reshape(3, 10)
    Y = np.arange(30).reshape(3, 10)
    number_of_batches = 3
    X_batches, Y_batches = NN.divide_batches(X, Y, number_of_batches)
    for i in range(len(X_batches)):
        assert np.all(X_batches[i] == Y_batches[i])
        assert X_batches[i].shape[0] == 3
        assert X_batches[i].shape[1] <= 2 * number_of_batches


def test_update_parameters():
    NN = NeuralNetwork(5)
    NN.add_hidden_layer(4, "RELU", "hidden_layer_1")
    NN.add_output_layer(2, "RELU", "MSE", "output_layer")
    NN.dW[1] = np.ones((4, 5))
    NN.db[1] = np.ones((4, 1))
    NN.dW[2] = np.ones((2, 4))
    NN.db[2] = np.ones((2, 1))
    old_weights_1 = NN.weights[1].copy()
    old_biases_1 = NN.biases[1].copy()
    old_weights_2 = NN.weights[2].copy()
    old_biases_2 = NN.biases[2].copy()
    NN.update_parameters(0.1)
    assert np.allclose(NN.weights[1], old_weights_1 - 0.1)
    assert np.allclose(NN.biases[1], old_biases_1 - 0.1)
    assert np.allclose(NN.weights[2], old_weights_2 - 0.1)
    assert np.allclose(NN.biases[2], old_biases_2 - 0.1)


def test_update_parameters_bad_lr():
    NN = NeuralNetwork(5)
    badvalues = [0, -0.1]
    for badvalue in badvalues:
        with raises(ValueError):
            NN.update_parameters(badvalue)
    badtypes = ["1", None, [1]]
    for badtype in badtypes:
        with raises(TypeError):
            NN.update_parameters(badtype)


def test_evaluate():
    NN = NeuralNetwork(5)
    NN.add_output_layer(2, "RELU", "MSE", "output_layer")
    Y = np.random.rand(10, 10)
    Yhat = Y
    assert NN.evaluate(Y, Yhat) == 0


def test_evaluate_incorrect_Y_type():
    NN = NeuralNetwork(5)
    Y = [1, 2]
    Yhat = np.array([[1, 1], [0, 0]])
    with raises(TypeError):
        NN.evaluate(Y, Yhat)


def test_evaluate_incorrect_Yhat_type():
    NN = NeuralNetwork(5)
    Y = np.array([[1, 1], [0, 0]])
    Yhat = [1, 2]
    with raises(TypeError):
        NN.evaluate(Y, Yhat)


def test_evaluate_Y_not_2d():
    NN = NeuralNetwork(5)
    Y = np.array([1, 2])
    Yhat = np.array([[1, 1], [0, 0]])
    with raises(ValueError):
        NN.evaluate(Y, Yhat)


def test_evaluate_Yhat_not_2d():
    NN = NeuralNetwork(5)
    Y = np.array([[1, 1], [0, 0]])
    Yhat = np.array([1, 2])
    with raises(ValueError):
        NN.evaluate(Y, Yhat)


def test_evaluate_Y_Yhat_shape_mismatch():
    NN = NeuralNetwork(5)
    Y = np.array([[1, 1], [0, 0]])
    Yhat = np.array([[1, 1], [0, 0], [1, 1]])
    with raises(ValueError):
        NN.evaluate(Y, Yhat)


def test_accuracy():
    NN = NeuralNetwork(5)
    NN.add_output_layer(2, "softmax", "cross_entropy", "output_layer")
    Y = np.array([[0, 1], [1, 0]])
    Yhat = np.array([[0, 1], [1, 0]])
    assert NN.accuracy(Y, Yhat) == 1


def test_accuracy_incorrect_Y_type():
    NN = NeuralNetwork(5)
    Y = [0, 1]
    Yhat = np.array([[0, 1], [1, 0]])
    with raises(TypeError):
        NN.accuracy(Y, Yhat)


def test_accuracy_incorrect_Yhat_type():
    NN = NeuralNetwork(5)
    Y = np.array([[0, 1], [1, 0]])
    Yhat = [0, 1]
    with raises(TypeError):
        NN.accuracy(Y, Yhat)


def test_accuracy_Y_not_2d():
    NN = NeuralNetwork(5)
    Y = np.array([0, 1])
    Yhat = np.array([[0, 1], [1, 0]])
    with raises(ValueError):
        NN.accuracy(Y, Yhat)


def test_accuracy_Yhat_not_2d():
    NN = NeuralNetwork(5)
    Y = np.array([[0, 1], [1, 0]])
    Yhat = np.array([0, 1])
    with raises(ValueError):
        NN.accuracy(Y, Yhat)


def test_accuracy_Y_Yhat_shape_mismatch():
    NN = NeuralNetwork(5)
    Y = np.array([[0, 1], [1, 0]])
    Yhat = np.array([[0, 1], [1, 0], [1, 0]])
    with raises(ValueError):
        NN.accuracy(Y, Yhat)


def test_train():
    NN = NeuralNetwork(3)
    NN.add_output_layer(1, "sigmoid", "MSE")
    X = np.random.rand(3, 5)
    Y = np.random.rand(1, 5)
    epochs = 3
    lr = 0.1
    batch_size = 2
    NN.train(X, Y, epochs, lr, batch_size)


def test_train_batch():
    NN = NeuralNetwork(3)
    NN.add_output_layer(1, "sigmoid", "MSE")
    X = np.random.rand(3, 5)
    Y = np.random.rand(1, 5)
    lr = 0.1
    NN.train_batch(X, Y, lr)


@pytest.mark.parametrize(
    "X,Y,epochs,lr,batch_size",
    [
        (
            np.random.rand(3, 5),
            np.random.rand(2, 5),
            3,
            0.1,
            2,
        ),  # Y rows don't match output layer
        (
            np.random.rand(3, 5),
            np.random.rand(1, 6),
            3,
            0.1,
            2,
        ),  # Y columns don't match X
        (3, np.random.rand(1, 5), 3, 0.1, 2),  # X is not numpy array
        (np.random.rand(3, 5), "Y", 3, 0.1, 2),  # Y is not numpy array
        (np.random.rand(6), np.random.rand(1, 5), 3, 0.1, 2),  # X is not 2D
        (np.random.rand(3, 5), np.random.rand(5), 3, 0.1, 2),  # Y is not 2D
        (
            np.random.rand(4, 5),
            np.random.rand(1, 5),
            3,
            0.1,
            2,
        ),  # X.shape[0] does not correspond to input layer
        (
            np.random.rand(3, 5),
            np.random.rand(1, 5),
            -1,
            0.1,
            2,
        ),  # epochs is not positive
        (
            np.random.rand(3, 5),
            np.random.rand(1, 5),
            3,
            -0.1,
            2,
        ),  # lr is not positive
        (
            np.random.rand(3, 5),
            np.random.rand(1, 5),
            3,
            0.1,
            0,
        ),  # batch_size is not positive
        (
            np.random.rand(3, 5),
            np.random.rand(1, 5),
            3,
            0.1,
            6,
        ),  # batch_size is larger than samples
    ],
)
def test_train_bad_input(X, Y, epochs, lr, batch_size):
    NN = NeuralNetwork(3)
    NN.add_output_layer(1, "sigmoid", "MSE")
    with raises((ValueError, TypeError)):
        NN.train(X, Y, epochs, lr, batch_size)


@pytest.mark.parametrize(
    "X,Y,lr",
    [
        (
            np.random.rand(3, 5),
            np.random.rand(2, 5),
            0.1,
        ),  # Y rows don't match output layer
        (
            np.random.rand(3, 5),
            np.random.rand(1, 6),
            0.1,
        ),  # Y columns don't match X
        (np.random.rand(6), np.random.rand(1, 5), 0.1),  # X is not 2D
        (np.random.rand(3, 5), np.random.rand(5), 0.1),  # Y is not 2D
        (
            np.random.rand(4, 5),
            np.random.rand(1, 5),
            0.1,
        ),  # X.shape[0] does not correspond to input layer
        (3, np.random.rand(1, 5), 0.1),  # X is not numpy array
        (np.random.rand(3, 5), "Y", 0.1),  # Y is not numpy array
        (
            np.random.rand(3, 5),
            np.random.rand(1, 5),
            -0.1,
        ),  # lr is not positive
    ],
)
def test_train_batch_bad_input(X, Y, lr):
    NN = NeuralNetwork(3)
    NN.add_output_layer(1, "sigmoid", "MSE")
    with raises((ValueError, TypeError)):
        NN.train_batch(X, Y, lr)
