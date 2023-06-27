import numpy as np


def activation_function(Z, name):
    if name == "sigmoid":
        return sigmoid(Z)
    elif name == "RELU":
        return RELU(Z)
    elif name == "tanh":
        return tanh(Z)
    elif name == "softmax":
        return softmax(Z)
    elif name == "identity":
        return identity(Z)


def derivative_activation_function(Z, name):
    if name == "sigmoid":
        return derivative_sigmoid(Z)
    elif name == "RELU":
        return derivative_RELU(Z)
    elif name == "tanh":
        return derivative_tanh(Z)
    elif name == "identity":
        return derivative_identity(Z)


def identity(Z):
    return Z


def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0)


def sigmoid(Z):
    return np.where(Z >= 0, 1 / (1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z)))


def RELU(Z):
    transformed_Z = np.zeros(np.shape(Z))
    transformed_Z[np.where(Z > 0)] = Z[np.where(Z > 0)]
    return transformed_Z


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def derivative_identity(Z):
    return np.ones(np.shape(Z))


def derivative_sigmoid(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


def derivative_RELU(Z):
    derivative = np.zeros(np.shape(Z))
    derivative[Z > 0] = 1
    return derivative


def derivative_tanh(Z):
    t = tanh(Z)
    return 1 - t**2
