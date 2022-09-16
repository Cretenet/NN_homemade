import numpy as np

def activation_function(matrix, name):
    if name == 'sigmoid' : return sigmoid(matrix)
    elif name == 'RELU' : return RELU(matrix)
    elif name == 'tanh' : return tanh(matrix)

def sigmoid(matrix):
    return 1/(1+np.exp(-matrix))

def RELU(matrix):
    transformed_matrix = np.zeros(np.shape(matrix))
    transformed_matrix[np.where(matrix > 0)] = matrix[np.where(matrix > 0)]
    return transformed_matrix

def tanh(matrix):
    return ((np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix)))