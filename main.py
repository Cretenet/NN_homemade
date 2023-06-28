import numpy as np

from neural_network import NeuralNetwork


def main():
    X_train, Y_train, X_test, Y_test = load_MNIST()
    NN = NeuralNetwork(input_size=784)
    NN.add_hidden_layer(nb_neurons=64, activation="RELU")
    NN.add_output_layer(
        cost="cross_entropy", nb_outputs=10, activation="softmax"
    )
    NN.train(
        X=X_train, Y=Y_train, epochs=10, lr=0.1, batch_size=32, eval_size=0.1
    )


def load_MNIST():
    # Load MNIST dataset
    X_train = np.load("MNIST/X_train.npy")
    X_test = np.load("MNIST/X_test.npy")
    Y_train = np.load("MNIST/Y_train.npy")
    Y_test = np.load("MNIST/Y_test.npy")
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    main()
