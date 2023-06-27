import numpy as np


def cost_function(Y, Yhat, name):
    """
    Computes the cost function for given true and predicted values.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.
    name : str
        Name of the cost function to be used.

    Returns
    -------
    float
        Computed cost using the specified cost function.
    """
    if name == "MSE":
        return MSE(Y, Yhat)
    elif name == "cross_entropy":
        return cross_entropy(Y, Yhat)
    elif name == "binary_cross_entropy":
        return binary_cross_entropy(Y, Yhat)


def derivative_cost_function(Y, Yhat, name):
    """
    Computes the derivative of the cost function for given
    true and predicted values.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.
    name : str
        Name of the cost function whose derivative is to be used.

    Returns
    -------
    numpy.ndarray
        Computed derivative using the specified cost function.
    """
    if name == "MSE":
        return derivative_MSE(Y, Yhat)
    elif name == "cross_entropy":
        return derivative_cross_entropy(Y, Yhat)
    elif name == "binary_cross_entropy":
        return derivative_binary_cross_entropy(Y, Yhat)


def MSE(Y, Yhat):
    """
    Computes the Mean Squared Error (MSE) cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    float
        Computed MSE cost.
    """
    losses = np.sum(np.square(Y - Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def cross_entropy(Y, Yhat):
    """
    Computes the cross-entropy cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    float
        Computed cross-entropy cost.
    """
    epsilon = 1e-15
    Yhat = np.clip(Yhat, epsilon, 1.0 - epsilon)
    losses = np.sum(-Y * np.log(Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def binary_cross_entropy(Y, Yhat):
    """
    Computes the binary cross-entropy cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    float
        Computed binary cross-entropy cost.
    """
    epsilon = 1e-15
    Yhat = np.clip(Yhat, epsilon, 1.0 - epsilon)
    losses = np.sum(-Y * np.log(Yhat) - (1 - Y) * np.log(1 - Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def derivative_MSE(Y, Yhat):
    """
    Computes the derivative of the Mean Squared Error (MSE) cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    numpy.ndarray
        Computed derivative of the MSE cost.
    """
    return 2 * (Yhat - Y)


def derivative_cross_entropy(Y, Yhat):
    """
    Computes the derivative of the cross-entropy cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    numpy.ndarray
        Computed derivative of the cross-entropy cost.
    """
    epsilon = 1e-15
    Yhat = np.clip(Yhat, epsilon, 1.0 - epsilon)
    return -Y / Yhat


def derivative_binary_cross_entropy(Y, Yhat):
    """
    Computes the derivative of the binary cross-entropy cost.

    Parameters
    ----------
    Y : numpy.ndarray
        True labels or target values.
    Yhat : numpy.ndarray
        Predicted labels or values.

    Returns
    -------
    numpy.ndarray
        Computed derivative of the binary cross-entropy cost.
    """
    epsilon = 1e-15
    Yhat = np.clip(Yhat, epsilon, 1.0 - epsilon)
    return -Y / Yhat + (1 - Y) / (1 - Yhat)
