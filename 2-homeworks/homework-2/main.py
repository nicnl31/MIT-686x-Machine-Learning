# CODE BASE FOR HOMEWORK 2

import numpy as np
from numpy import nan


# Problem 1 codes:
def squared_error(Y, X):
    """
    Compute the squared error term of the empirical risk function.

    :param Y: The original matrix of ratings, which may consist of NaN values.
    :param X: The predicted matrix.
    :return: The squared error of Y and X.
    """
    res = 0
    for a in range(Y.shape[0]):
        for i in range(Y.shape[1]):
            if not np.isnan(Y[a][i]):
                res += (Y[a][i] - X[a][i])**2
    return res/2


def regularisation(U, V, k, l):
    """
    Compute the regularisation term.
    :param U: Matrix U.
    :param V: Matrix V.
    :param k: rank of the predicted matrix X.
    :param l: lambda, the regularisation parameter.
    :return:
    """
    u_sum, v_sum = 0, 0
    for a in range(len(U)):
        for j in range(k):
            u_sum += U[a][j]**2
    for i in range(len(V)):
        for j in range(k):
            v_sum += V[i][j]**2
    return 0.5*l*(u_sum + v_sum)


# Problem statement: You are given the following matrices Y_0, and U_0, V_0.
Y_0 = np.array([[5, nan, 7],
                [nan, 2, nan],
                [4, nan, nan],
                [nan, 3, 6]])
U_0 = np.array([[6, 0, 3, 6]]).T
V_0 = np.array([[4, 2, 1]]).T


# Problem 1a: Initialise matrices U_0 and V_0, and compute the matrix X_0
X_0 = np.matmul(U_0, V_0.T)
print('--------------------')
print('    Problem 1a:')
print(f'X_0 = \n {X_0}')


# Problem 1b: Compute the squared error term
se = squared_error(Y_0, X_0)
reg = regularisation(U_0, V_0, 1, 1)
print('--------------------')
print('    Problem 1b:')
print(f'Squared error: {se}')
print(f'Regularisation: {reg}')
