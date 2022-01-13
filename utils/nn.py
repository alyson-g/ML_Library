# Functions used in neural networks
import numpy as np
from scipy.special import expit


def softmax(x):
    """
    Calculate the softmax for the given values
    :param x: A matrix of values
    :return A calculated softmax matrix
    """
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(r, y):
    """
    Calculate cross-entropy error
    :param r: The actual values
    :param y: The predicted values
    :return: A cross-entropy value
    """
    return -np.sum(np.sum(r * np.log(y)))


def sigmoid(x):
    """
    Calculate the sigmoid of a value
    :param x: An np.array
    :return: The calculated value
    """
    return expit(x)
