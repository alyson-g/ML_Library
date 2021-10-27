# Defines the k-nearest neighbor classifier algorithm
from collections import Counter

import numpy as np

from utils.cross_validation import k_fold
from utils.distance import euclidean_distance
from utils.evaluate import classification_score
from utils.plot import plot
from utils.preprocessing import drop_column


class KNearestClassifier:
    def __init__(self, k):
        self.k = k
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

    def edited_fit(self, X, y):
        """
        Fit the model to data
        :param X: A DataFrame to fit the model to
        :param y: A list of class values for each row in X
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
                distances = euclidean_distance(Z, [X.iloc[index]])

                # Sort by nearest and take the first 1
                sorted_indices = np.argsort(distances.flatten())
                nearest_class = Z_y.iloc[sorted_indices[0]]

                # If class was predicted correctly, remove it
                if nearest_class == y.iloc[index]:
                    indices.remove(index)
                    changed = True

        self.X = X.iloc[indices]
        self.y = y.iloc[indices]

    def condensed_fit(self, X, y):
        """
        Fit the model to data
        :param X: A DataFrame to fit the model to
        :param y: A list of class values for each row in X
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
                nearest_class = y.iloc[sorted_indices[0]]

                # If class was predicted incorrectly, add it
                if not nearest_class == y.iloc[index]:
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
        classes = []

        for index in range(X.shape[0]):
            # Find the distance between the point and all stored points
            distances = euclidean_distance(self.X, [X.iloc[index]])

            # Sort by nearest and take the first k
            sorted_indices = np.argsort(distances.flatten())
            nearest = list(self.y.iloc[sorted_indices[:self.k]])

            # Find most common of the k-nearest neighbors - the predicted class label
            cls = Counter(np.array(nearest)).most_common(1)[0][0]

            # Demo output
            if index == 0 and self.verbose:
                print(f"Point classification: {cls}")

            classes.append(cls)

        return classes


def find_best_k_classification(data, target, k_values, fit_type, display_graph=False):
    """
    Find the best k value
    :param data: A DataFrame
    :param target: The target column
    :param k_values: k values to test
    :param fit_type: Either 'edited', 'condensed', or 'regular'
    :param display_graph: Whether or not to display a graph of outputs
    :return: The best k value
    """
    if fit_type not in ['edited', 'condensed', 'regular']:
        raise 'type parameter must be either edited, condensed, or regular'

    folds = k_fold(data, 5, validation=False)

    scores = []

    for k in k_values:
        _scores = []

        for fold in folds:
            train = fold['train']
            test = fold['test']

            knn = KNearestClassifier(k)
            X = drop_column(train, target)
            y = train[target]

            if fit_type == 'edited':
                knn.edited_fit(X, y)
            elif fit_type == 'condensed':
                knn.condensed_fit(X, y)
            else:
                knn.fit(X, y)

            predicted = knn.predict(drop_column(test, target))
            score = classification_score(predicted, test[target].tolist())

            _scores.append(score)

        avg_score = np.mean(_scores)
        scores.append(avg_score)

    errors = [1 - score for score in scores]

    # Display elbow graph for visual inspection
    if display_graph:
        plot(k_values, errors, 'k', 'Error Rate')

    # Return the best k value
    return k_values[errors.index(min(errors))]
