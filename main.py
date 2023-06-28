from tensorflow.keras import datasets, utils

from neural_network import NeuralNetwork


def main():
    X_train, Y_train, X_test, Y_test = load_MNIST()
    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train[:][:3])
    NN = NeuralNetwork(input_size=784)
    NN.add_hidden_layer(nb_neurons=128, activation="RELU")
    NN.add_hidden_layer(nb_neurons=64, activation="RELU")
    NN.add_output_layer(
        cost="cross_entropy", nb_outputs=10, activation="softmax"
    )
    NN.train(X=X_train, Y=Y_train, epochs=10, lr=0.1, batch_size=32)


def load_MNIST():
    # Load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()

    # Normalize the images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten the images
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # One-hot encode the labels
    Y_train = utils.to_categorical(Y_train)
    Y_test = utils.to_categorical(Y_test)

    return X_train.T, Y_train.T, X_test.T, Y_test.T


if __name__ == "__main__":
    main()
