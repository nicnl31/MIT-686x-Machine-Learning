import numpy as np
np.set_printoptions(formatter={'float_kind': '{:.15f}'.format})


# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM 1: Neural Networks
class NeuralNetwork(object):
    def __init__(self, input_to_hidden_matrix, hidden_to_output_matrix, x_train, activation, output_activation):
        self.input_to_hidden_matrix = input_to_hidden_matrix
        self.hidden_to_output_matrix = hidden_to_output_matrix

        self.x_train = np.append(x_train, [1])

        self.activation = activation
        self.output_activation = output_activation

    def train(self, beta):
        pre_activation_1 = self.input_to_hidden_matrix.T @ self.x_train
        activation_1 = np.append(self.activation(pre_activation_1), [1])
        pre_activation_2 = self.hidden_to_output_matrix.T @ activation_1
        activation_2 = self.activation(pre_activation_2)

        output = self.output_activation(activation_2, beta)

        return output.reshape((len(output), ))

    def get_activation_values(self):
        pre_activation_1 = self.input_to_hidden_matrix.T @ self.x_train
        activation_1 = np.append(self.activation(pre_activation_1), [1])
        pre_activation_2 = self.hidden_to_output_matrix.T @ activation_1
        activation_2 = self.activation(pre_activation_2)

        return activation_2


def ReLU(z):
    """
    Computes the rectified linear unit (ReLU) activation function.
    Args:
        z: The pre-activation array of shape (m, ).

    Returns:
        ReLU(z): the rectified linear unit activation function output of z.
    """
    zero = np.zeros(len(z))
    return np.maximum(zero, z)


def softmax(u, beta):
    """
    Computes the softmax activation function of the output unit.
    Args:
        u: The pre-activation array of shape (m, ).

    Returns:
        The softmax probabilities of array u.

    """
    return np.exp(beta * u) / np.sum(np.exp(beta * u), axis=0)


W1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, -1, -1, -1]])  # Includes offset W_0
V1 = np.array([[1, -1], [1, -1], [1, -1], [1, -1], [0, 2]])  # Includes offset V_0
x_train = np.array([3, 14])
nn_1 = NeuralNetwork(W1, V1, x_train, ReLU, softmax)

# ----- Part a: Feed-forward step
train_output = nn_1.train(beta=1)

# TODO: Uncomment below to print answer
# print(f"Training output: {train_output}")


# ----- Part b: Inverse temperature
activations = nn_1.get_activation_values()


def answer(beta):
    output = softmax(activations, beta)
    while output[1] < 1e-3:
        activations[0] -= 1e-4
        output = softmax(activations, beta)
    print(f"Difference between f(u_1) and f(u_2) (beta = {beta}): {activations[0] - activations[1]}")

# TODO: Uncomment below to print answer
# for beta in [1, 3]:
#     answer(beta)


# ----------------------------------------------------------------------------------------------------------------------
# PROBLEM 2: Long Short Term Memory RNN
def sigmoid(z):
    """
    Computes the sigmoid activation function.
    Args:
        z: The pre-activation array of shape (m, ).

    Returns:
        The sigmoid activation function output of z.

    """
    return 1/(1 + np.exp(-z))


class LSTM(object):
    def __init__(self):
        self.memory_cell = []
        self.hidden = []

    def set_forget_state(self, state, x_train, w_hidden, w_input, f_bias):
        pre_activation = w_hidden @ self.hidden
        return sigmoid(pre_activation)
