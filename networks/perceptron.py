# Perceptron regression algorithm
import numpy as np

from utils.evaluate import mean_squared_error


class Perceptron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.w = None

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

        # Initialize w
        self.w = np.random.uniform(-0.01, 0.01, num_cols + 1)

        # Loop through each example until convergence has been reached
        convergence_reached = False

        max_iterations = 1000
        iterations = 0

        indices = np.array(range(num_rows))
        while not convergence_reached and iterations < max_iterations:
            np.random.shuffle(indices)

            for i in indices:
                output = np.dot(self.w.T, _X[i])
                self.w = self.learning_rate * (_y[i] - output) * _X[i]

            outputs = np.empty(num_rows)
            for i in range(num_rows):
                outputs[i] = np.dot(self.w.T, _X[i])

            error = mean_squared_error(outputs, _y[indices])

            # Check if convergence has been reached
            if error == 0:
                convergence_reached = True

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

        predicted = []

        for i in range(len(_X)):
            predicted.append(np.dot(self.w.T, _X[i]))

        return predicted
