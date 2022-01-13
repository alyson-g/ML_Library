# Functions used in neural networks
import numpy as np


def softmax(x):
    """
    Calculate the softmax for the given values
    :param x: A matrix of values
    :return A calculated softmax matrix
    """
    return np.exp(x) / np.sum(np.exp(x))
