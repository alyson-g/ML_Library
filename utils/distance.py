# Distance functions
from scipy.spatial import distance


def euclidean_distance(x, y):
    """
    Calculates the Euclidean distance between two points or arrays of points
    :param x: A point or array of points
    :param y: A point or array of points
    :return: The Euclidean distance
    """
    return distance.cdist(x, y, metric='euclidean')

def hamming_distance(x, y):
    """
    Calculates the Hamming distance between two points or arrays of points
    :param x: A point or array of points
    :param y: A point or array of points
    :return: The Hamming distance
    """
    return distance.cdist(x, y, metric='hamming')


def minkowski_distance(x, y):
    """
        Calculates the Minkowski distance between two points or arrays of points
        :param x: A point or array of points
        :param y: A point or array of points
        :return: The Minkowski distance
        """
    return distance.cdist(x, y, metric='minkowski')


def gaussian_kernel(distances, sigma=2):
    """
    Calculate the Gaussian kernel
    :param distances: A vector of distances
    :param sigma: The bandwidth parameter
    :return: The Gaussian kernel
    """
    return np.exp(-(1 / 2 * sigma) * distances)