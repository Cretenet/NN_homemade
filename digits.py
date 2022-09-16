from neural_network import neural_network
import numpy as np

NN=neural_network(input_size=2, input_layer_name='input')
NN.add_hidden_layer("first hidden", 1, activation_function='RELU')
NN.add_hidden_layer("second hidden", 3, activation_function='RELU')
NN.add_output_layer("output",2,activation_function='sigmoid')
NN.info()
print(NN.forward_propagation(np.array([1,1])))