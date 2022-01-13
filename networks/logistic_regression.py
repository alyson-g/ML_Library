# Logistic regression class
import numpy as np
import pandas as pd

from utils.cross_validation import k_fold
from utils.evaluate import classification_score
from utils.nn import softmax
from utils.preprocessing import drop_column, one_hot_encode, standardize


class LogisticRegression:
    def __init__(self, learning_rate, max_iterations=None):
        self.learning_rate = learning_rate
        self.w = None
        self.classes = []

        if max_iterations is None:
            self.max_iterations = np.inf
        else:
            self.max_iterations = max_iterations

    def fit(self, X, y):
        """
        Fit the model
        :param X: A DataFrame of features
        :param y: A DataFrame of output values
        :return: None
        """
        num_rows = X.shape[0]
        num_cols = X.shape[1]

        # Prepend bias input to each row
        _X = np.array(X)
        ones = np.ones((_X.shape[0], 1))
        _X = np.concatenate((ones, _X), axis=1)

        _y = np.array(y)

        self.classes = list(y)

        # Initialize weights to random values
        k = len(list(y))
        self.w = np.random.uniform(-0.01, 0.01, (k, num_cols + 1))

        convergence_reached = False
        iterations = 0

        # Iterate through all examples until weight convergence is reached
        while not convergence_reached and iterations < self.max_iterations:
            delta_w = np.zeros_like(self.w)

            for n in range(num_rows):
                o = np.empty(k)

                for j in range(k):
                    o[j] = np.dot(self.w[j].T, _X[n])

                y_hat = softmax(o)
                r = np.array(y.iloc[n])

                for i in range(k):
                    for j in range(num_cols):
                        delta_w[i][j] = self.learning_rate * (delta_w[i][j] + (r[i] - y_hat[i]) * _X[n, j])

            # Recalculate weight matrix
            temp_w = self.w + delta_w

            # Calculate cross-entropy
            error = 0

            for i in range(num_rows):
                o = np.empty(k)
                for j in range(k):
                    o[j] = np.dot(temp_w[j].T, _X[i])

                y_hat = np.nan_to_num(softmax(o))
                error = error + np.sum(_y[i] * np.log(y_hat))

            error = -error

            # Check if cross-entropy has converged
            if error == 0:
                convergence_reached = True
            else:
                self.w = temp_w

            iterations = iterations + 1

    def predict(self, X):
        """
        Predict classifications for the given examples
        :param X: A DataFrame of features
        :return: Predicted values
        """
        # Prepend bias input to each row
        _X = np.array(X)
        ones = np.ones((_X.shape[0], 1))
        _X = np.concatenate((ones, _X), axis=1)

        predictions = []
        for n in range(len(_X)):
            vals = np.empty(len(self.w))
            for i in range(len(self.w)):
                vals[i] = np.dot(self.w[i].T, _X[n])

            index = np.argmax(vals)
            predictions.append(self.classes[index])

        return predictions


def find_best_learning_rate(data, target, values):
    """
    Find the best learning rate
    :param data: A DataFrame
    :param target: The target column
    :param values: The values to test
    :return: The best learning rate
    """
    folds = k_fold(data, 5, validation=False)

    best_score = 0
    best_learning_rate = None

    for value in values:
        for fold in folds:
            train = fold['train']
            test = fold['test']

            train_X = drop_column(train, target)
            test_X = drop_column(test, target)

            # Standardize the data
            train_X, test_X = standardize(train_X, test_X)

            # One-hot encode target class
            train_y = pd.DataFrame(train[target], columns=[target])
            train_y = one_hot_encode(train_y, target)
            test_y = np.array(test[target])

            classifier = LogisticRegression(value)
            classifier.fit(train_X, train_y)

            predicted = classifier.predict(test_X)

            # Calculate classification score
            predicted = [x.replace(f"{target}_", '') for x in predicted]
            test_y = [str(x) for x in test_y]
            score = classification_score(predicted, test_y)

            if score > best_score:
                best_score = score
                best_learning_rate = value

    return best_learning_rate
