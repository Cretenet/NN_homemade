import numpy as np
import activation_functions as af

class neural_network:
    """
    The neural_network class allows the user to create, train, and use an artificial neural network.
    The layers must be created in the right order, the input layer is automatically created.
    The user can create n hidden layers by calling n times add_hidden_layer().
    The user must create an output layer as last layer with add_output_layer().
    """

    def __init__(self, input_size, input_layer_name='input_layer'):
       self.nbLayers = 1
       self.layers_size = [input_size]
       self.input_size = input_size
       self.names = [input_layer_name]
       self.choice_activation_functions = []
       self.output_layer_exists = False
       self.weights = {}
       self.biases = {}
       self.activation = {}
       self.derivative_activation = {}
       self.dell_delb = {} # Gradient of the loss function w.r.t the biases
       self.dell_delw = {} # Gradient of the loss function w.r.t the weights

    def add_hidden_layer(self, name, nb_neurons=1, activation_function = 'RELU'):
        if not self.output_layer_exists:
            self.nbLayers += 1
            self.layers_size.append(nb_neurons)
            self.names.append(name)
            self.initialize_weights(activation_function, self.nbLayers-2, self.layers_size[self.nbLayers-2], nb_neurons)
            if activation_function =='RELU' or activation_function == 'sigmoid' or activation_function=='tanh':
                self.choice_activation_functions.append(activation_function)
            else :
                self.choice_activation_functions.append('RELU')
                print('The chosen activation function is not supported, RELU is used instead.')
        else :
            print('It is not possible to add a hidden layer after the output layer.' \
            'The layers must be added in the right order.')

    def add_output_layer(self, name='output_layer', nb_outputs=1, activation_function = 'sigmoid'):
        if self.nbLayers == 1:
            print('There must be at least one hidden layer before putting the output layer.')
        else :
            self.output_layer_exists = True
            self.nbLayers += 1
            self.layers_size.append(nb_outputs)
            self.names.append(name)
            self.initialize_weights(activation_function, self.nbLayers-2, self.layers_size[self.nbLayers-2], nb_outputs)
            if activation_function =='RELU' or activation_function == 'sigmoid' or activation_function=='tanh':
                self.choice_activation_functions.append(activation_function)
            else :
                self.choice_activation_functions.append('sigmoid')
                print('The chosen activation function is not supported, sigmoid is used instead.')

    def initialize_weights(self, activation_function, index, nb_first_layer, nb_second_layer):
        if activation_function == 'RELU': 
            m=2
        elif activation_function == 'tanh' or activation_function=='sigmoid':
            m=1
        self.weights[index] = np.random.normal(0,m/self.input_size,(nb_second_layer,nb_first_layer))
        self.biases[index] = np.zeros((nb_second_layer,1))

    def forward_propagation(self, input):
        first_layer=np.reshape(input,(len(input),1))
        self.activation[0] = first_layer
        for i in range(0,len(self.names)-1) :
            W = self.weights[i]
            second_layer = af.activation_function(W.dot(first_layer)+ self.biases[i], self.choice_activation_functions[i])
            # The following two lines are useful for backward propagation
            self.activation[i+1] = second_layer
            self.derivative_activation[i] = af.derivative_activation_function(W.dot(first_layer)+ self.biases[i], self.choice_activation_functions[i])
            first_layer = second_layer # Restart with the next one
        return second_layer
       
 
    def evaluate(self, prediction, truth):
        prediction = np.reshape(prediction,(len(prediction),1))
        truth = np.reshape(truth,(len(truth),1))
        return np.sum(np.square(prediction-truth)) # Sum of squares

    def backward_propagation(self, truth):
        truth = np.reshape(truth,(len(truth),1))
        index_left_layer = self.nbLayers-2
        index_right_layer = self.nbLayers-1
        left_layer_size = self.layers_size[index_left_layer]
        right_layer_size = self.layers_size[index_right_layer]
        self.dell_delw[self.nbLayers-2] = np.zeros((right_layer_size,left_layer_size))
        self.dell_delw[self.nbLayers-2][0][0] = 2 * (self.activation[self.nbLayers-1][0]-truth[0]) * self.derivative_activation[self.nbLayers-2][0]*self.activation[self.nbLayers-2][0]
        self.dell_delw[self.nbLayers-2][1][0] = 2 * (self.activation[self.nbLayers-1][1]-truth[1]) * self.derivative_activation[self.nbLayers-2][1]*self.activation[self.nbLayers-2][0]
        previous_dell_dela = 2* (self.activation[self.nbLayers-1]-truth)
        for l in range(0,self.nbLayers-1): # Loop over the layers
            index_left_layer = self.nbLayers-2-l
            index_right_layer = self.nbLayers-1-l
            left_layer_size = self.layers_size[index_left_layer]
            right_layer_size = self.layers_size[index_right_layer]
            self.dell_delw[self.nbLayers-2-l] = np.zeros((right_layer_size,left_layer_size))
            self.dell_delb[self.nbLayers-2-l] = np.zeros((right_layer_size,1))
            for j in range(0,right_layer_size):
                self.dell_delb[self.nbLayers-2-l][j][0]=previous_dell_dela[j] * self.derivative_activation[self.nbLayers-2-l][j]
                for k in range(0,left_layer_size):
                    self.dell_delw[self.nbLayers-2-l][j][k] = previous_dell_dela[j] * self.derivative_activation[self.nbLayers-2-l][j]*self.activation[self.nbLayers-2-l][k]
            dell_dela=np.zeros((left_layer_size,1))
            for k in range(0,left_layer_size):
                for j in range(0,right_layer_size):
                    dell_dela[k][0] += self.weights[self.nbLayers-2-l][j][k]*self.derivative_activation[self.nbLayers-2-l][j]*previous_dell_dela[j]
            previous_dell_dela = dell_dela

    def info(self):
        message = 'The fully connected artificial neural network is composed of :\n'
        message += '\t - An input layer named \"'+str(self.names[0])+'\" expecting '+ str(self.layers_size[0])+ \
            ' inputs.'
        if self.nbLayers > 1:
            if self.output_layer_exists:
                for i in range(1,self.nbLayers-1): 
                    message += '\n\t - A hidden layer named \"'+str(self.names[i])+'\", containing '+ \
                        str(self.layers_size[i])+' neurons, and using the '+str(self.choice_activation_functions[i-1])+ \
                        ' activation function.'
                message += '\n\t - An output layer named \"'+str(self.names[-1])+'\", containing '+ \
                    str(self.layers_size[-1])+' neurons, and using the '+str(self.choice_activation_functions[-1])+\
                    ' activation function.'
            else :
                for i in range(1,self.nbLayers): 
                    message += '\n\t - A hidden layer named \"'+str(self.names[i])+'\", containing '+\
                        str(self.layers_size[i])+' neurons, and using the '+str(self.choice_activation_functions[i-1])+\
                        ' activation function.'
        message += '\n Now let us look at the different weight matrices :\n'
        print(message)
        for i in range(1,len(self.names)) :
            print('W' + str(i)+' = ')
            print(self.weights[i-1])
            print('\n')