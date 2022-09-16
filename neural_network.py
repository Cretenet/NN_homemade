from random import weibullvariate
from unicodedata import name
import numpy as np
import activation_functions as af

class neural_network:

    def __init__(self, input_size, input_layer_name='input_layer'):
       self.L = 1 # Number of layers
       self.layers_size = [input_size] # Size of each layer
       self.input_size = input_size
       self.names = [input_layer_name] # Name of each layer
       self.activation_functions = [] # Activation function of each layer
       self.output_layer_exists = False # Is there already an output layer?
       self.weights = {} # Weights of each new layer
       self.biases = np.array([])

    def add_hidden_layer(self, name, nb_neurons=1, activation_function = 'RELU'):
        if not self.output_layer_exists:
            self.L += 1
            self.layers_size.append(nb_neurons)
            self.names.append(name)
            self.initialize_weights(activation_function, self.L-2, self.layers_size[self.L-2], nb_neurons)
            if activation_function =='RELU' or activation_function == 'sigmoid' or activation_function=='tanh':
                self.activation_functions.append(activation_function)
            else :
                self.activation_functions.append('RELU')
                print('The chosen activation function is not supported, RELU is used instead.')
        else :
            print('It is not possible to add a hidden layer after the output layer.' \
            'The layers must be added in the right order.')

    def add_output_layer(self, name='output_layer', nb_outputs=1, activation_function = 'sigmoid'):
        if self.L == 1:
            print('There must be at least one hidden layer before putting the output layer.')
        else :
            self.output_layer_exists = True
            self.L += 1
            self.layers_size.append(nb_outputs)
            self.names.append(name)
            self.initialize_weights(activation_function, self.L-2, self.layers_size[self.L-2], nb_outputs)
            if activation_function =='RELU' or activation_function == 'sigmoid' or activation_function=='tanh':
                self.activation_functions.append(activation_function)
            else :
                self.activation_functions.append('sigmoid')
                print('The chosen activation function is not supported, sigmoid is used instead.')

    def initialize_weights(self, activation_function, index, nb_first_layer, nb_second_layer):
        if activation_function == 'RELU': 
            m=2
        elif activation_function == 'tanh' or activation_function=='sigmoid':
            m=1
        self.weights[index] = np.random.normal(0,m/self.input_size,(nb_second_layer,nb_first_layer))
        self.biases = np.append(self.biases, [0])

    def forward_propagation(self, input):
        first_layer=np.reshape(input,(len(input),1))
        for i in range(0,len(self.names)-1) :
            W = self.weights[i]
            second_layer = af.activation_function(W.dot(first_layer)+ self.biases[i], self.activation_functions[i])
            print(second_layer)
            first_layer = second_layer
        return second_layer

    def evaluate(self, prediction, truth):
        return np.sum(np.square(prediction-truth)) # Sum of squares

    def info(self):
        message = 'The fully connected artificial neural network is composed of :\n'
        message += '\t - An input layer named \"'+str(self.names[0])+'\" expecting '+ str(self.layers_size[0])+ \
            ' inputs.'
        if self.L > 1:
            if self.output_layer_exists:
                for i in range(1,self.L-1): 
                    message += '\n\t - A hidden layer named \"'+str(self.names[i])+'\", containing '+ \
                        str(self.layers_size[i])+' neurons, and using the '+str(self.activation_functions[i-1])+ \
                        ' activation function.'
                message += '\n\t - An output layer named \"'+str(self.names[-1])+'\", containing '+ \
                    str(self.layers_size[-1])+' neurons, and using the '+str(self.activation_functions[-1])+\
                    ' activation function.'
            else :
                for i in range(1,self.L): 
                    message += '\n\t - A hidden layer named \"'+str(self.names[i])+'\", containing '+\
                        str(self.layers_size[i])+' neurons, and using the '+str(self.activation_functions[i-1])+\
                        ' activation function.'
        message += '\n Now let us look at the different weight matrices :\n'
        print(message)
        for i in range(1,len(self.names)) :
            print('W' + str(i)+' = ')
            print(self.weights[i-1])
            print('\n')