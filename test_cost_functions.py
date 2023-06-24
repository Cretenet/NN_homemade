import numpy as np

from cost_functions import (
    MSE,
    binary_cross_entropy,
    cost_function,
    cross_entropy,
    derivative_binary_cross_entropy,
    derivative_cost_function,
    derivative_cross_entropy,
    derivative_MSE,
)


def test_MSE():
    Y = np.array([[0.5, 1.0]])
    Yhat = np.array([[0.6, 0.9]])
    assert np.isclose(MSE(Y, Yhat), 0.01)
    assert np.isclose(cost_function(Y, Yhat, "MSE"), 0.01)


def test_cross_entropy():
    Y = np.array([[1, 0]])
    Yhat = np.array([[0.6, 0.4]])
    assert np.isclose(cross_entropy(Y, Yhat), 0.25541281188)
    assert np.isclose(cost_function(Y, Yhat, "cross_entropy"), 0.25541281188)


def test_binary_cross_entropy():
    Y = np.array([[1, 0]])
    Yhat = np.array([[0.6, 0.4]])
    assert np.isclose(binary_cross_entropy(Y, Yhat), 0.51082562376)
    assert np.isclose(
        cost_function(Y, Yhat, "binary_cross_entropy"), 0.51082562376
    )


def test_derivative_MSE():
    epsilon = 1e-10
    Yhat = np.array([[0.5, 1.0]])
    Y = np.array([[0.6, 0.9]])
    Yhat_plus_1 = np.array([[0.5 + epsilon, 1.0]])
    Yhat_minus_1 = np.array([[0.5 - epsilon, 1.0]])
    derivative_1 = (MSE(Y, Yhat_plus_1) - MSE(Y, Yhat_minus_1)) / (2 * epsilon)
    Yhat_plus_2 = np.array([[0.5, 1.0 + epsilon]])
    Yhat_minus_2 = np.array([[0.5, 1.0 - epsilon]])
    derivative_2 = (MSE(Y, Yhat_plus_2) - MSE(Y, Yhat_minus_2)) / (2 * epsilon)
    assert np.allclose(
        derivative_MSE(Y, Yhat), np.array([[derivative_1, derivative_2]])
    )
    assert np.allclose(
        derivative_cost_function(Y, Yhat, "MSE"), derivative_MSE(Y, Yhat)
    )


def test_derivative_cross_entropy():
    epsilon = 1e-10
    Y = np.array([[1, 0]])
    Yhat = np.array([[0.6, 0.4]])
    Yhat_plus_1 = np.array([[0.6 + epsilon, 0.4]])
    Yhat_minus_1 = np.array([[0.6 - epsilon, 0.4]])
    derivative_1 = (
        cross_entropy(Y, Yhat_plus_1) - cross_entropy(Y, Yhat_minus_1)
    ) / (2 * epsilon)
    Yhat_plus_2 = np.array([[0.6, 0.4 + epsilon]])
    Yhat_minus_2 = np.array([[0.6, 0.4 - epsilon]])
    derivative_2 = (
        cross_entropy(Y, Yhat_plus_2) - cross_entropy(Y, Yhat_minus_2)
    ) / (2 * epsilon)
    assert np.allclose(
        derivative_cross_entropy(Y, Yhat),
        np.array([[derivative_1, derivative_2]]),
    )
    assert np.allclose(
        derivative_cost_function(Y, Yhat, "cross_entropy"),
        derivative_cross_entropy(Y, Yhat),
    )


def test_derivative_binary_cross_entropy():
    epsilon = 1e-10
    Y = np.array([[1, 0]])
    Yhat = np.array([[0.6, 0.4]])
    Yhat_plus_1 = np.array([[0.6 + epsilon, 0.4]])
    Yhat_minus_1 = np.array([[0.6 - epsilon, 0.4]])
    derivative_1 = (
        binary_cross_entropy(Y, Yhat_plus_1)
        - binary_cross_entropy(Y, Yhat_minus_1)
    ) / (2 * epsilon)
    Yhat_plus_2 = np.array([[0.6, 0.4 + epsilon]])
    Yhat_minus_2 = np.array([[0.6, 0.4 - epsilon]])
    derivative_2 = (
        binary_cross_entropy(Y, Yhat_plus_2)
        - binary_cross_entropy(Y, Yhat_minus_2)
    ) / (2 * epsilon)
    assert np.allclose(
        derivative_binary_cross_entropy(Y, Yhat),
        np.array([[derivative_1, derivative_2]]),
    )
    assert np.allclose(
        derivative_cost_function(Y, Yhat, "binary_cross_entropy"),
        derivative_binary_cross_entropy(Y, Yhat),
    )
