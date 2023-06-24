import numpy as np


def cost_function(Y, Yhat, name):
    if name == "MSE":
        return MSE(Y, Yhat)
    elif name == "cross_entropy":
        return cross_entropy(Y, Yhat)
    elif name == "binary_cross_entropy":
        return binary_cross_entropy(Y, Yhat)


def derivative_cost_function(Y, Yhat, name):
    if name == "MSE":
        return derivative_MSE(Y, Yhat)
    elif name == "cross_entropy":
        return derivative_cross_entropy(Y, Yhat)
    elif name == "binary_cross_entropy":
        return derivative_binary_cross_entropy(Y, Yhat)


def MSE(Y, Yhat):
    losses = np.sum(np.square(Y - Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def cross_entropy(Y, Yhat):
    losses = np.sum(-Y * np.log(Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def binary_cross_entropy(Y, Yhat):
    losses = np.sum(-Y * np.log(Yhat) - (1 - Y) * np.log(1 - Yhat), axis=0)
    return np.sum(losses) / Y.shape[1]


def derivative_MSE(Y, Yhat):
    return 2 * (Yhat - Y) / Y.shape[1]


def derivative_cross_entropy(Y, Yhat):
    return -Y / Yhat / Y.shape[1]


def derivative_binary_cross_entropy(Y, Yhat):
    return -Y / Yhat / Y.shape[1] + (1 - Y) / (1 - Yhat) / Y.shape[1]
