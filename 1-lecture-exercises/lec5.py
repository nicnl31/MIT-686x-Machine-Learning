import numpy as np


def loss_single(feature_vector, label, theta, loss=None):
    """
    Finds the loss on a single data point given specific regression parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        loss - The loss function type.

    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    z = label - np.dot(theta, feature_vector)  # z is the argument of the Hinge loss function
    if loss == 'hinge':
        if z >= 1:
            return 0
        return 1-z
    elif loss == 'squared_error':
        return 0.5*z**2


def loss_full(feature_matrix, labels, theta, loss):
    """
    Finds the total specified loss on a set of data given specific parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        loss - The loss function to utilise

    Returns: A real number representing the specified loss value associated with the
    given dataset and parameters. This number should be the average
    loss across all points in the feature matrix.
    """
    loss = [loss_single(feature_matrix[i], labels[i], theta, loss=loss) for i in range(len(feature_matrix))]
    return sum(loss)/len(loss)


# Problem 1-2: compute hinge loss and squared error loss on some training data
x_train = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, -1],
    [-1, 1, 1]
])
y_train = [2, 2.7, -.7, 2]
params = np.array([0, 1, 2])

# Hinge loss
print(loss_full(x_train, y_train, params, loss='hinge'))
# Squared error loss
print(loss_full(x_train, y_train, params, loss='squared_error'))
