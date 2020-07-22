import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class MLP:
    def __init__(self, size=None):
        """
        :param size:[784,30,10]
        """
        if size is None:
            size = [784, 30, 10]
        self.size = size
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(ch) for ch in size[1:]]

    def forward(self, x):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            x = sigmoid(z)
        return x

    def backward(self, x, y):
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        activations = [x]
        zs = []
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        return


def main():
    pass


if __name__ == '__main__':
    main()
