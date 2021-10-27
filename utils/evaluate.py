# Evaluation metrics
import numpy as np


# Classification metrics ----------------------------------------------------------------------------------------------
def classification_score(predicted, actual):
    """
    Calculate a classification accuracy score
    :param predicted: A list of predicted values
    :param actual: A list of actual values
    :return: An accuracy score
    """
    # Ensure lists are of equal length
    if not len(predicted) == len(actual):
        raise 'predicted and actual must be the same length'

    n = len(predicted)
    return np.sum(np.square(np.subtract(actual, predicted))) / n


# Regression metrics --------------------------------------------------------------------------------------------------
def mean_squared_error(predicted, actual):
    """
    Calculate the mean squared error
    :param predicted: A list of predicted values
    :param actual: A list of actual values
    :return: The mean squared error
    """
    # Ensure lists are of equal length
    if not len(predicted) == len(actual):
        raise 'predicted and actual must be the same length'

    n = len(predicted)
    error = 0

    for index in range(len(predicted)):
        error = error + (actual[index] - predicted[index]) ** 2

    return error / n


def r_squared(predicted, actual):
    """
    Calculate r squared
    :param predicted:
    :param actual:
    :return: R squared
    """
    # Ensure lists are of equal length
    if not len(predicted) == len(actual):
        raise 'predicted and actual must be the same length'

    rss = 0
    tss = 0

    y_hat = np.mean(actual)

    for index in range(len(predicted)):
        rss = rss + (actual[index] - predicted[index]) ** 2
        tss = tss + (actual[index] - y_hat) ** 2

    return 1 - (rss / tss)


def pearson(x, y):
    """
    Calculate the Pearson Correlation Coefficient
    :param x: The x variable
    :param y: The y variable
    :return: r, the correlation coefficient
    """
    # Ensure lists are of equal length
    if not len(x) == len(y):
        raise 'x and y must be the same length'

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = 0
    denom_x = 0
    denom_y = 0

    for index in range(len(x)):
        numerator = numerator + (x[index] - mean_x) * (y[index] - mean_y)
        denom_x = denom_x + (x[index] - mean_x) ** 2
        denom_y = denom_y + (y[index] - mean_y) ** 2

    return numerator / (denom_x * denom_x) ** 0.5
