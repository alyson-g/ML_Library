# Multilayer Perceptron regression algorithm
import numpy as np

from utils.evaluate import mean_squared_error
from utils.nn import sigmoid


class MLPRegression:
    def __init__(self, learning_rate, hidden_layer_sizes, max_iter=100):
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.w = None
        self.v = None
        self.max_iter = max_iter

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

        # Initialize weight matrices
        self.w = np.random.uniform(-0.01, 0.01, (self.hidden_layer_sizes[0], num_cols + 1))

        self.v = np.empty(len(self.hidden_layer_sizes), dtype=object)
        for i in range(len(self.hidden_layer_sizes) + 1):
            if i == 0:
                continue

            if i == len(self.hidden_layer_sizes):
                self.v[i - 1] = np.random.uniform(-0.01, 0.01, (1, self.hidden_layer_sizes[i - 1] + 1))
                self.v[i - 1] = self.v[i - 1].flatten()
            else:
                self.v[i - 1] = np.random.uniform(-0.01, 0.01, (self.hidden_layer_sizes[i],
                                                                self.hidden_layer_sizes[i - 1] + 1))

        # Loop through examples until convergence is reached
        convergence_reached = False

        indices = np.array(range(num_rows))
        epochs = 0

        while not convergence_reached and epochs <= self.max_iter:
            # Shuffle data for stochastic gradient descent
            np.random.shuffle(indices)

            for i in indices:
                z = self.calculate_output(_X[i], self.w, self.v)

                r = _y[i]
                err = [r - z[len(z) - 1]]

                # Back propagate through hidden layers
                for j in reversed(range(len(self.v))):
                    # Update error for previous layer before changing weights
                    next_error = np.zeros_like(z[j])

                    for k in range(len(err)):
                        for m in range(len(self.v[j])):
                            next_error = next_error + err[k] * self.v[j][m]

                    # Calculate weight change
                    if j == len(self.v) - 1:
                        for k in range(len(self.v[j])):
                            delta = self.learning_rate * np.sum(err[0] * z[j])
                            self.v[j][k] = self.v[j][k] + delta
                    else:
                        for k in range(len(self.v[j])):
                            delta = self.learning_rate * z[j] * np.sum(err[k] * z[j + 1] * (1 - z[j + 1]))
                            self.v[j][k] = self.v[j][k] + delta

                    err = next_error

                # Back propagate through first layer
                for j in range(len(self.w)):
                    delta = self.learning_rate * np.sum(err[j] * z[0] * (1 - z[0])) * _X[i]
                    self.w[j] = self.w[j] + delta

            # Calculate predicted values after all weight updates
            predicted = np.empty(len(indices))
            for i in range(len(indices)):
                predicted[i] = self.calculate_output(_X[indices[i]], self.w, self.v)[-1]

            error = mean_squared_error(predicted, _y)

            if error == 0:
                convergence_reached = True

            epochs = epochs + 1

    def calculate_output(self, x, w, v):
        """
        Calculate the output given a data point and set of weights
        :param x: A data point
        :param w: A weight matrix for the first layer
        :param v: A weight matrix for the hidden layers
        :return: An output matrix
        """
        z = np.empty(len(self.hidden_layer_sizes) + 1, dtype=object)
        len_v = len(v)
        len_z = len(z)

        # Calculate output into first hidden layer
        z[0] = sigmoid(np.tensordot(w.T, x, axes=([0], [0])))
        z[0] = np.concatenate((np.ones(1), z[0]), axis=0)

        # Iterate through each hidden layer
        for j in range(len_v - 1):
            z[j + 1] = sigmoid(np.tensordot(v[j], z[j], axes=1))
            z[j + 1] = np.concatenate((np.ones(1), z[j + 1]), axis=0)

        # Calculate final output
        z[len_z - 1] = np.dot(v[len_v - 1].T, z[len_z - 2])

        return z

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

        predicted = np.empty(len(_X))
        for i in range(len(_X)):
            predicted[i] = self.calculate_output(_X[i], self.w, self.v)[-1]

        return predicted
