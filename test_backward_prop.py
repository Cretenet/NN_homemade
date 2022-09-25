from neural_network import neural_network
import numpy as np

np.random.seed(0)
epsilon = 1e-6
NN=neural_network(input_size=2, input_layer_name='input')
NN.add_hidden_layer("first hidden", 1, activation_function='sigmoid')
NN.add_hidden_layer("second hidden", 3, activation_function='sigmoid')
NN.add_output_layer("output",2,activation_function='sigmoid')
v1 = NN.forward_propagation(np.array([0.5,0.7]))
l1 = NN.evaluate(v1,[0.5,0.4])

NN.backward_propagation(np.array([0.5,0.4]))
ok = True
for l in range(0,NN.L-1):
    for i in range(0,NN.weights[l].shape[0]):
        for j in range(0,NN.weights[l].shape[1]):
            NN.weights[l][i][j]+= epsilon
            v2 = NN.forward_propagation(np.array([0.5,0.7]))
            l2 = NN.evaluate(v2,[0.5,0.4])
            derivative = (l2-l1)/epsilon
            NN.weights[l][i][j]-= epsilon
            # Compute the difference (small if the code works)
            error = np.abs(derivative-NN.dell_delw[l][i][j])
            if error > epsilon : 
                ok = False
                print('The backward propagation gives a different result than the finite difference method'\
                    +' for the weight at the index (l,i,j) = ('+str(l)+', '+str(i)+', '+str(j)+')')
                
if ok: print('The backward propagation gives the same result as using finite difference for the weights (up to'\
    ' an error of '+str(epsilon)+').')

ok = True
for l in range(0,NN.L-1):
    for i in range(0,len(NN.dell_delb[l])):
        NN.biases[l][i] += epsilon
        v2 = NN.forward_propagation(np.array([0.5,0.7]))
        l2 = NN.evaluate(v2,[0.5,0.4])
        derivative = (l2-l1)/epsilon
        NN.biases[l][i] -= epsilon # Reset the change we made on the i-th bias
        error = np.abs(derivative-NN.dell_delb[l][i])
        if error > epsilon : 
            ok = False
            print('The backward propagation gives a different result than the finite difference method'\
                +' for the bias at the index (l,i,j) = ('+str(l)+', '+str(i)+', '+str(j)+')')

if ok: print('The backward propagation gives the same result as using finite difference for the biases (up to'\
    ' an error of '+str(epsilon)+').')