import numpy as np

from neural_network import neural_network

np.random.seed(0)
NN = neural_network(input_size=2, input_layer_name="input")
NN.add_hidden_layer("first hidden", 1, activation_function="sigmoid")
NN.add_hidden_layer("second hidden", 3, activation_function="sigmoid")
NN.add_output_layer("output", 2, activation_function="sigmoid")
print("NN INFO :")
NN.info()
print("output :")
print(NN.forward_propagation(np.array([0.5, 0.7])))
NN.backward_propagation(np.array([[0.5], [0.4]]))
