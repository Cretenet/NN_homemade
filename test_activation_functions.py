import numpy as np

from activation_functions import (
    RELU,
    activation_function,
    derivative_activation_function,
    derivative_identity,
    derivative_RELU,
    derivative_sigmoid,
    derivative_tanh,
    identity,
    sigmoid,
    softmax,
    tanh,
)


def test_identity():
    assert np.allclose(
        identity(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]])
    )
    assert np.allclose(
        activation_function(np.array([[1, 2], [3, 4]]), "identity"),
        identity(np.array([[1, 2], [3, 4]])),
    )


def test_softmax():
    assert np.allclose(
        softmax(np.array([[1, 2], [3, 4]])),
        np.array(
            [[0.11920292202, 0.11920292202], [0.88079707797, 0.88079707797]]
        ),
    )
    assert np.allclose(
        activation_function(np.array([[1, 2], [3, 4]]), "softmax"),
        softmax(np.array([[1, 2], [3, 4]])),
    )


def test_sigmoid():
    assert np.allclose(
        sigmoid(np.array([[1, 2], [3, 4]])),
        np.array(
            [[0.73105857863, 0.88079707797], [0.95257412682, 0.98201379004]]
        ),
    )
    assert np.allclose(
        activation_function(np.array([[1, 2], [3, 4]]), "sigmoid"),
        sigmoid(np.array([[1, 2], [3, 4]])),
    )


def test_RELU():
    assert np.allclose(
        RELU(np.array([[-1, 2], [3, -4]])), np.array([[0, 2], [3, 0]])
    )
    assert np.allclose(
        activation_function(np.array([[1, 2], [3, 4]]), "RELU"),
        RELU(np.array([[1, 2], [3, 4]])),
    )


def test_tanh():
    assert np.allclose(
        tanh(np.array([[1, 2], [3, 4]])),
        np.array(
            [[0.76159415595, 0.96402758007], [0.99505475369, 0.99932929974]]
        ),
    )
    assert np.allclose(
        activation_function(np.array([[1, 2], [3, 4]]), "tanh"),
        tanh(np.array([[1, 2], [3, 4]])),
    )


def test_derivative_identity():
    epsilon = 1e-10
    matrix = np.random.rand(10, 10)
    derivative_plus = identity(matrix + epsilon)
    derivative_minus = identity(matrix - epsilon)
    derivative = (derivative_plus - derivative_minus) / (2 * epsilon)
    assert np.allclose(derivative_identity(matrix), derivative)
    assert np.allclose(
        derivative_activation_function(matrix, "identity"),
        derivative_identity(matrix),
    )


def test_derivative_sigmoid():
    epsilon = 1e-10
    matrix = np.random.rand(10, 10)
    derivative_plus = sigmoid(matrix + epsilon)
    derivative_minus = sigmoid(matrix - epsilon)
    derivative = (derivative_plus - derivative_minus) / (2 * epsilon)
    print(derivative - derivative_sigmoid(matrix))
    assert np.allclose(derivative_sigmoid(matrix), derivative)
    assert np.allclose(
        derivative_activation_function((matrix), "sigmoid"),
        derivative_sigmoid(matrix),
    )


def test_derivative_RELU():
    epsilon = 1e-10
    matrix = np.random.rand(10, 10)
    derivative_plus = RELU(matrix + epsilon)
    derivative_minus = RELU(matrix - epsilon)
    derivative = (derivative_plus - derivative_minus) / (2 * epsilon)
    assert np.allclose(derivative_RELU(matrix), derivative)
    assert np.allclose(
        derivative_activation_function(matrix, "RELU"), derivative_RELU(matrix)
    )


def test_derivative_tanh():
    epsilon = 1e-10
    matrix = np.random.rand(10, 10)
    derivative_plus = tanh(matrix + epsilon)
    derivative_minus = tanh(matrix - epsilon)
    derivative = (derivative_plus - derivative_minus) / (2 * epsilon)
    assert np.allclose(derivative_tanh(matrix), derivative)
    assert np.allclose(
        derivative_activation_function(matrix, "tanh"), derivative_tanh(matrix)
    )
