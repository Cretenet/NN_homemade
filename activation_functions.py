import numpy as np

def activation_function(matrix, name):
    if name == 'sigmoid' : return sigmoid(matrix)
    elif name == 'RELU' : return RELU(matrix)
    elif name == 'tanh' : return tanh(matrix)

def derivative_activation_function(matrix,name):
    if name == 'sigmoid' : return derivative_sigmoid(matrix)
    elif name == 'RELU' : return derivative_RELU(matrix)
    elif name == 'tanh' : return derivative_tanh(matrix)

def sigmoid(matrix):
    return 1.0/(1+np.exp(-matrix))

def RELU(matrix):
    transformed_matrix = np.zeros(np.shape(matrix))
    transformed_matrix[np.where(matrix > 0)] = matrix[np.where(matrix > 0)]
    return transformed_matrix

def tanh(matrix):
    return ((np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix)))

def derivative_sigmoid(matrix):
    return np.exp(-matrix)/(1+np.exp(-matrix))/(1+np.exp(-matrix))

def derivative_RELU(matrix):
    derivative = np.zeros(np.shape(matrix))
    derivative[np.where(matrix > 0)] = np.ones(np.shape(derivative[np.where(matrix > 0)]))
    return derivative

def derivative_tanh(matrix):
    return 1.0/(np.cosh(matrix)*np.cosh(matrix))
