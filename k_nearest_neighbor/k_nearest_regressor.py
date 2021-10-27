# Defines the k-nearest neighbor regression algorithm
import numpy as np

from utils.cross_validation import k_fold
from utils.distance import euclidean_distance, gaussian_kernel
from utils.evaluate import mean_squared_error
from utils.plot import plot
from utils.preprocessing import drop_column


class KNearestRegressor:
    def __init__(self, k, sigma=2):
        self.k = k
        self.sigma = sigma
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fit the model to data
        :param X: A DataFrame to fit the model to
        :param y: A list of class values for each row in X
        :return: None
        """
        self.X = X
        self.y = y

    def edited_fit(self, X, y, epsilon):
        """
        Fit the model to data using the Edited KNN method
        :param X: A DataFrame to fit the model to
        :param y: A list of class values for each row in X
        :parameter epsilon: The error threshold
        :return: None
        """
        indices = list(range(X.shape[0]))

        # Shuffle data set
        X = X.iloc[indices]
        y = y.iloc[indices]

        # Variable to indicate if there has been a change in Z
        changed = True

        while changed:
            changed = False

            for index in range(X.shape[0]):
                # If value has already been removed, continue
                if index not in indices:
                    continue

                # Find the distance between the point and all stored points
                temp_indices = indices.copy()
                temp_indices.remove(index)

                Z = X.iloc[temp_indices]
                Z_y = y.iloc[temp_indices]

                # Find the distance between the point and all stored points
                distances = euclidean_distance(Z, [X.iloc[index]])

                # Sort by nearest and take the first k
                sorted_indices = np.argsort(distances.flatten())
                nearest = Z_y.iloc[sorted_indices[0]]
                weight = gaussian_kernel(nearest, self.sigma)

                # Calculate regression value
                prediction = weight * nearest / weight

                # If class was predicted correctly, remove it
                if np.abs(prediction - y.iloc[index]) <= epsilon:
                    indices.remove(index)
                    changed = True

        self.X = X.iloc[indices]
        self.y = y.iloc[indices]

    def condensed_fit(self, X, y, epsilon):
        """
        Fit the model to data
        :param X: A DataFrame to fit the model to
        :param y: A list of class values for each row in X
        :parameter epsilon: The error threshold
        :return: None
        """
        indices = []

        # Shuffle data set
        shuffled = list(range(X.shape[0]))
        X = X.iloc[shuffled]
        y = y.iloc[shuffled]

        # Variable to indicate if there has been a change in Z
        changed = True

        while changed:
            changed = False

            for index in range(X.shape[0]):
                # If Z is empty, add first element
                if len(indices) == 0:
                    indices.append(index)
                    changed = True
                    continue

                # If x is already in Z, continue
                if index in indices:
                    continue

                # Find the distance between the point and all stored points
                Z = X.iloc[indices]
                distances = euclidean_distance(Z, [X.iloc[index]])

                # Sort by nearest and take the first 1
                sorted_indices = np.argsort(distances.flatten())
                nearest = y.iloc[sorted_indices[0]]
                weight = gaussian_kernel(nearest, self.sigma)

                # Calculate regression value
                prediction = weight * nearest / weight

                # If class was predicted incorrectly, add it
                if not np.abs(prediction - y.iloc[index]) <= epsilon:
                    indices.append(index)
                    changed = True

        self.X = X.iloc[indices]
        self.y = y.iloc[indices]

    def predict(self, X):
        """
        Predict the class of given data points
        :param X: A list of data points to predict
        :return: A list of predicted values
        """
        predictions = []

        for index in range(X.shape[0]):
            # Find the distance between the point and all stored points
            distances = euclidean_distance(self.X, [X.iloc[index]])

            # Sort by nearest and take the first k
            sorted_indices = np.argsort(distances.flatten())
            nearest_distances = distances[sorted_indices[:self.k]].flatten()
            weights = gaussian_kernel(nearest_distances, self.sigma)

            # Calculate regression value
            nearest = list(self.y.iloc[sorted_indices[:self.k]])
            predictions.append(sum(weights * nearest) / sum(weights))

        return predictions


def find_best_k_regression(data, target, k_values, sigma=2, display_graph=False):
    """
    Find the best k value
    :param data: A DataFrame
    :param target: The target column
    :param k_values: k values to test
    :param sigma: The sigma value to pass to the Gaussian kernel
    :param display_graph: Whether or not to display a graph of outputs
    :return: The best k value
    """
    folds = k_fold(data, 5, validation=False)

    errors = []

    for k in k_values:
        _errors = []

        for fold in folds:
            train = fold['train']
            test = fold['test']

            knn = KNearestRegressor(k, sigma=sigma)
            X = drop_column(train, target)
            y = train[target]

            knn.fit(X, y)

            predicted = knn.predict(drop_column(test, target))
            error = mean_squared_error(predicted, test[target].tolist())

            _errors.append(error)

        avg_error = np.mean(_errors)
        errors.append(avg_error)

    # Display elbow graph for visual inspection
    if display_graph:
        plot(k_values, errors, 'k', 'Error Rate')

    # Return the best k value
    return k_values[errors.index(min(errors))]


def find_best_sigma(data, target, k, sigma_values, display_graph=False):
    """
    Find the best k value
    :param data: A DataFrame
    :param target: The target column
    :param k: k nearest neighbors
    :param sigma_values: Sigma values to test
    :param display_graph: Whether or not to display a graph of outputs
    :return: The best k value
    """
    folds = k_fold(data, 5, validation=False)

    errors = []

    for sigma in sigma_values:
        _errors = []

        for fold in folds:
            train = fold['train']
            test = fold['test']

            knn = KNearestRegressor(k, sigma=sigma)
            X = drop_column(train, target)
            y = train[target]

            knn.fit(X, y)

            predicted = knn.predict(drop_column(test, target))
            error = mean_squared_error(predicted, test[target].tolist())

            _errors.append(error)

        avg_error = np.mean(_errors)
        errors.append(avg_error)

    # Display elbow graph for visual inspection
    if display_graph:
        plot(sigma_values, errors, 'Sigma', 'Error Rate')

    # Return the best k value
    return sigma_values[errors.index(min(errors))]


def find_best_epsilon(data, target, k, epsilon_values, fit_type, sigma=2, display_graph=False):
    """
    Find the best epsilon value
    :param data: A DataFrame
    :param target: The target column
    :param k: K nearest neighbors to use
    :param epsilon_values: Epsilon values to test
    :param fit_type: Either 'edited' or 'condensed'
    :param sigma: The sigma value to pass to the Gaussian kernel
    :param display_graph: Whether or not to display a graph of outputs
    :return: The best epsilon value
    """
    if fit_type not in ['edited', 'condensed']:
        raise 'type parameter must be either edited or condensed'

    folds = k_fold(data, 5, validation=False)

    errors = []

    for e in epsilon_values:
        _errors = []

        for fold in folds:
            train = fold['train']
            test = fold['test']

            knn = KNearestRegressor(k, sigma=sigma)
            X = drop_column(train, target)
            y = train[target]

            if fit_type == 'edited':
                knn.edited_fit(X, y, e)
            elif fit_type == 'condensed':
                knn.condensed_fit(X, y, e)

            predicted = knn.predict(drop_column(test, target))
            error = mean_squared_error(predicted, test[target].tolist())

            _errors.append(error)

        avg_error = np.mean(_errors)
        errors.append(avg_error)

    # Display elbow graph for visual inspection
    if display_graph:
        plot(epsilon_values, errors, 'e', 'Error Rate')

    # Return the best k value
    return epsilon_values[errors.index(min(errors))]
