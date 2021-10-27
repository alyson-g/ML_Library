# Data plotting functions
import matplotlib.pyplot as plt


def plot(x, y, x_label, y_label):
    """
    Plot a graph
    :param x: X-axis parameter
    :param y: Y-axis parameter
    :param x_label: X-axis label
    :param y_label: Y-axis label
    :return: None
    """
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, 'bx-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
