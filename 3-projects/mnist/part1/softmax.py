import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sys
sys.path.append("..")
import utils
from mnist.utils import *


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n data points each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    H_temp = np.matmul(theta, X.T)/temp_parameter  # returns a (k, n) matrix before exponentiation
    c = np.max(H_temp, axis=0)  # constant to subtract to minimise numerical overflow
    H = np.exp(H_temp - c)  # exp and subtract c
    H /= np.sum(H, axis=0)  # divide by sum to get probabilities

    return H


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n data points each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    n = X.shape[0]  # Number of training examples
    k = theta.shape[0]  # Number of classes
    ln_p = np.log(compute_probabilities(X, theta, temp_parameter))
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()

    loss = (-1/n) * np.sum(ln_p[M == 1])
    reg = (lambda_factor / 2) * np.linalg.norm(theta) ** 2

    return loss + reg


# Test case
# X = np.array([[ 1., 60., 70., 79., 45., 88., 71., 83., 18., 26., 26.],
#  [1., 39., 88.,  5., 21., 59., 84., 17., 99., 74., 14.],
#  [1., 16., 12., 43., 44., 77., 69., 10., 11., 74., 35.],
#  [1., 89., 70., 71., 63., 98., 94., 65., 27., 33., 22.],
#  [1., 39., 26., 73., 78., 41., 58., 43., 17., 81., 97.],
#  [1., 75., 90., 42., 16., 82., 16., 94., 27., 14., 10.],
#  [1., 36., 65.,  8.,  4., 84., 35., 18., 92., 67., 26.],
#  [1., 38., 58., 71., 23., 64., 53., 75., 88., 31., 52.],
#  [1., 68., 32., 42., 10., 56., 90., 61., 47., 56.,  2.],
#  [1., 47., 41.,  8.,  2.,  6., 68., 48., 50., 72., 84.]])
# theta = np.array([[-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [ 0.27,   8.775,  8.235, 11.799, 10.503, 14.742, 12.096, 10.53,  14.364, 17.469, 15.525],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725],
#  [-0.03,  -0.975, -0.915, -1.311, -1.167, -1.638, -1.344, -1.17,  -1.596, -1.941, -1.725]])
# temp_parameter = 1.0
# lambda_factor = 0.0001
# Y = np.ones(10)
# print(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
# print(compute_probabilities(X, theta, temp_parameter))


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n = X.shape[0]  # Number of training examples
    k = theta.shape[0]  # Number of classes
    prob_matrix = compute_probabilities(X, theta, temp_parameter)  # (k, n) probability matrix
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()

    gradient = (-1/(temp_parameter * n)) * np.matmul((M - prob_matrix), X) + lambda_factor * theta  # gradient update
    theta -= alpha * gradient

    return theta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return train_y % 3, test_y % 3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    assigned_labels = get_classification(X, theta, temp_parameter) % 3
    return 1 - np.mean(assigned_labels == Y)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis=0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
