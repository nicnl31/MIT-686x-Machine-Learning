import numpy as np


# Hidden layer models

def Z(W, X, W_0):
    """
    Computes the pre-activation function, Z.
    Args:
        weights: (m, n) matrix of weights
        inputs: (m, ) vector
        offset: (n, ) vector

    Returns:
        Z: (n, ) vector

    """
    X_trans = np.array(X).reshape((len(X), 1))
    W_0_trans = W_0.reshape((len(W_0), 1))
    Z = W.T @ X_trans + W_0_trans

    return Z.reshape((len(Z), ))


def linear(Z):
    return 5*Z - 2


def ReLU(Z):
    zero = np.zeros(len(Z))
    return np.maximum(zero, Z)


def tanh(Z):
    return np.tanh(Z)


# Problem 0: Problem data
x_train = [[-1, -1], [1, -1], [-1,1], [1, 1]]
y_train = [1, -1, -1, 1]

# Problem 1: Linear Separability After First Layer
W_1 = np.array([[1, -1], [-1, 1]])
W_01 = np.array([1, 1])

for i in range(len(x_train)):
    print(f'Data point {i+1}, class {y_train[i]}, tanh activation: {tanh(Z(W_1, x_train[i], W_01))}')
