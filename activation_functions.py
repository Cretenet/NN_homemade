import numpy as np


def activation_function(matrix, name):
    if name == "sigmoid":
        return sigmoid(matrix)
    elif name == "RELU":
        return RELU(matrix)
    elif name == "tanh":
        return tanh(matrix)
    elif name == "softmax":
        return softmax(matrix)
    elif name == "identity":
        return identity(matrix)


def derivative_activation_function(matrix, name):
    if name == "sigmoid":
        return derivative_sigmoid(matrix)
    elif name == "RELU":
        return derivative_RELU(matrix)
    elif name == "tanh":
        return derivative_tanh(matrix)
    elif name == "identity":
        return derivative_identity(matrix)


def identity(matrix):
    return matrix


def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=0)


def sigmoid(matrix):
    return 1.0 / (1 + np.exp(-matrix))


def RELU(matrix):
    transformed_matrix = np.zeros(np.shape(matrix))
    transformed_matrix[np.where(matrix > 0)] = matrix[np.where(matrix > 0)]
    return transformed_matrix


def tanh(matrix):
    return (np.exp(matrix) - np.exp(-matrix)) / (
        np.exp(matrix) + np.exp(-matrix)
    )


def derivative_identity(matrix):
    return np.ones(np.shape(matrix))


def derivative_sigmoid(matrix):
    return np.exp(-matrix) / (1 + np.exp(-matrix)) / (1 + np.exp(-matrix))


def derivative_RELU(matrix):
    derivative = np.zeros(np.shape(matrix))
    derivative[np.where(matrix > 0)] = 1
    return derivative


def derivative_tanh(matrix):
    return 1.0 / (np.cosh(matrix) * np.cosh(matrix))
