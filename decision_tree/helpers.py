# Decision tree helper functions
from collections import Counter

import numpy as np


def entropy(data):
    """
    Calculate entropy
    :param data: Data to be calculated
    :return: Entropy calculation
    """
    values = np.unique(data)

    total = len(data)
    i = 0

    for val in values:
        p_i = len(list(filter(lambda x: x == val, data))) / total
        i = i + (p_i * np.log(p_i))

    return -i


def intrinsic_value(subsets, total):
    """
    Calculate the intrinsic value
    :param subsets: The split sets of data
    :param total: The total number of data points
    :return: Calculated intrinsic value
    """
    iv = 0

    for subset in subsets:
        iv = iv + (len(subset) / total) * np.log(len(subset) / total)

    return -iv


def most_frequent_class(data):
    """
    Find the most frequent class
    :param data: A list
    :return: The most frequent class
    """
    count = Counter(data)
    return count.most_common(1)[0][0]
