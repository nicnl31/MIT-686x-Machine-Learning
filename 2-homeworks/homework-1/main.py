import numpy as np
import matplotlib.pyplot as plt


# PROBLEM 1: Perceptron mistakes
def perceptron(x_train, y_train, offset=True, t=1):
    """
    Performs a perceptron algorithm and returns parameters theta (and theta_0 if offset=True)

    :param x_train: list of lists, each mini-list being a data point of the form [x_1, x_2]
    :param y_train: list of scalars, either 1 or -1
    :param offset: decides if the perceptron algorithm returns a hyperplane through the origin
    :param t: number of iterations

    :return:
        - If not offset, return theta
        - If offset, return theta and theta_0

    """
    theta = np.zeros(len(x_train[0]))
    mistakes = 0
    if not offset:
        for t in range(t):
            for i in range(len(x_train)):
                if y_train[i]*(np.dot(theta, np.array(x_train[i]))) <= 0:
                    theta = theta + y_train[i]*np.array(x_train[i])
                    mistakes += 1
                print(f'Iteration {i}: theta = {theta}')
        return theta, mistakes
    elif offset:
        theta_0 = 0
        for t in range(t):
            for i in range(len(x_train)):
                if y_train[i]*(np.dot(theta, np.array(x_train[i])) + theta_0) <= 0:
                    theta = theta + y_train[i]*np.array(x_train[i])
                    theta_0 += y_train[i]
                    mistakes += 1
                print(f'Iteration {i}: theta = {theta}, theta_0 = {theta_0}')
        return theta, theta_0, mistakes


# x_train_1 = [[-1, -1], [1, 0], [-1, 10]]
# y_train_1 = [1, -1, 1]
#
# x_train_2 = [[1, 0], [-1, 10], [-1, -1]]
# y_train_2 = [-1, 1, 1]
#
# print(perceptron(x_train_1, y_train_1, False, 20))
# print(perceptron(x_train_2, y_train_2, False, 1))

# Plotting
# x_train_3 = [[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]]
# y_train_3 = [1, 1, -1, -1, -1]
#
# theta, theta_0, mistakes = perceptron(x_train_3, y_train_3, True, 10)
# theta_3 = [i[0]*theta[0] + i[1]*theta[1] + theta_0 for i in x_train_3]
# for i in range(len(x_train_3)):
#     plt.scatter(x_train_3[i][0], x_train_3[i][1])
# plt.plot(theta_3)
# plt.show()

# x_train_4 = [[-1, 1], [1, -1], [1, 1], [2, 2]]
# y_train_4 = [1, 1, -1, -1]
# print(perceptron(x_train_4, y_train_4, True, 5))

x_train_5 = [[np.cos(np.pi), 0], [0, np.cos(2*np.pi)], [1, 1]]
y_train_5 = [1, 1, 1]
print(perceptron(x_train_5, y_train_5, False, 5))


# PROBLEM 2: Perceptron performance
def perceptron_with_mistakes(x_train, y_train, misclassified, offset=True):
    """
    Performs a perceptron algorithm and returns parameters theta (and theta_0 if offset=True)

    :param x_train: list of lists, each mini-list being a data point of the form [x_1, x_2]
    :param y_train: list of scalars, either 1 or -1
    :param misclassified: target mis-classification array
    :param offset: decides if the perceptron algorithm returns a hyperplane through the origin

    :return:
        - If not offset, return theta
        - If offset, return theta and theta_0

    """
    theta = np.zeros(len(x_train[0]))
    mistakes = np.zeros(len(x_train))
    if not offset:
        while not np.array_equal(mistakes, misclassified):
            for i in range(len(x_train)):
                if y_train[i]*(np.dot(theta, np.array(x_train[i]))) <= 0:
                    theta = theta + y_train[i]*np.array(x_train[i])
                    mistakes[i] += 1
                print(f'Iteration {i}: theta = {theta}')
        return theta
    elif offset:
        theta_0 = 0
        while not np.array_equal(mistakes, misclassified):
            for i in range(len(x_train)):
                if y_train[i]*(np.dot(theta, np.array(x_train[i])) + theta_0) <= 0:
                    theta = theta + y_train[i]*np.array(x_train[i])
                    theta_0 += y_train[i]
                    mistakes[i] += 1
                print(f'Iteration {i}: theta = {theta}, theta_0 = {theta_0}')
        return theta, theta_0


# 2a
# x_train_3 = [[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]]
# y_train_3 = [1, 1, -1, -1, -1]
# mis_classifications = np.array([1, 0, 2, 1, 0])
#
# print(perceptron_with_mistakes(x_train_3, y_train_3, mis_classifications))
