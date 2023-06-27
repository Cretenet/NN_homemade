import numpy as np


def activation_function(Z, name):
    """
    Selects the appropriate activation function and applies it to the input.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.
    name : str
        Name of the activation function to be used.

    Returns
    -------
    numpy.ndarray
        Activated values.
    """
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
    """
    Computes the derivative of the chosen activation function and
    applies it to the input.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.
    name : str
        Name of the activation function whose derivative is to be used.

    Returns
    -------
    numpy.ndarray
        The derivative of the activated values.
    """
    if name == "sigmoid":
        return derivative_sigmoid(Z)
    elif name == "RELU":
        return derivative_RELU(Z)
    elif name == "tanh":
        return derivative_tanh(Z)
    elif name == "identity":
        return derivative_identity(Z)


def identity(Z):
    """
    Identity activation function. It doesn't transform the input.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        The same input values (since identity function doesn't
        alter the input).
    """
    return Z


def softmax(Z):
    """
    Softmax activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        Activated values using softmax function.
    """
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0)


def sigmoid(Z):
    """
    Sigmoid activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        Activated values using sigmoid function.
    """
    return np.where(Z >= 0, 1 / (1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z)))


def RELU(Z):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        Activated values using ReLU function.
    """
    transformed_Z = np.zeros(np.shape(Z))
    transformed_Z[np.where(Z > 0)] = Z[np.where(Z > 0)]
    return transformed_Z


def tanh(Z):
    """
    Hyperbolic tangent (tanh) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        Activated values using tanh function.
    """
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def derivative_identity(Z):
    """
    Computes the derivative of the identity function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        The derivative of the identity function.
    """
    return np.ones(np.shape(Z))


def derivative_sigmoid(Z):
    """
    Computes the derivative of the sigmoid function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        The derivative of the sigmoid function.
    """
    sig = sigmoid(Z)
    return sig * (1 - sig)


def derivative_RELU(Z):
    """
    Computes the derivative of the ReLU function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        The derivative of the ReLU function.
    """
    derivative = np.zeros(np.shape(Z))
    derivative[Z > 0] = 1
    return derivative


def derivative_tanh(Z):
    """
    Computes the derivative of the tanh function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values at a given layer.

    Returns
    -------
    numpy.ndarray
        The derivative of the tanh function.
    """
    t = tanh(Z)
    return 1 - t**2
