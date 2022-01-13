# Autoencoder classification algorithm
import numpy as np

from utils.nn import cross_entropy, sigmoid, softmax


class AutoencoderClassification:
    def __init__(self, learning_rate, ae_layer, cls_layer, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # The number of units in the autoencoder hidden layer
        self.ae_layer = ae_layer

        # The number of units in the class prediction hidden layer
        self.cls_layer = cls_layer

        # Weights trained on autoencoder
        self.w_a = None

        # Weights trained on classification output
        self.w_c = None

        # Final weight matrix
        self.w = None

        self.classes = None

    def fit(self, X, y):
        """
        Fit the model
        :param X: A DataFrame of features
        :param y: A DataFrame of output values
        :return: None
        """
        self.classes = list(y)

        # Fit the autoencoder layer
        self.fit_autoencoder(X)

        num_rows = X.shape[0]

        # Prepend bias input to each row
        _X = np.array(X)
        ones = np.ones((_X.shape[0], 1))
        _X = np.concatenate((ones, _X), axis=1)

        _y = np.array(y)

        # Initialize weight matrices
        self.w_c = np.empty(2, dtype=object)

        self.w_c[0] = np.random.uniform(-0.01, 0.01, (self.cls_layer, self.ae_layer + 1))
        self.w_c[1] = np.random.uniform(-0.01, 0.01, (len(self.classes), self.cls_layer + 1))

        # Loop through examples until convergence is reached
        convergence_reached = False

        indices = np.array(range(num_rows))
        epochs = 0

        w = np.concatenate((self.w_a, self.w_c), axis=0)

        # Remove autoencoder output unit
        w = np.delete(w, 1, axis=0)

        while (not convergence_reached) and epochs <= self.max_iter:
            # Shuffle data for stochastic gradient descent
            np.random.shuffle(indices)

            for i in indices:
                z = self.calculate_output(_X[i], w)

                r = _y[i]
                error = r - z[-1]

                # Back propagate through hidden layers
                for j in reversed(range(1, len(w))):
                    # Update error for previous layer before changing weights
                    next_error = np.zeros_like(z[j - 1])
                    for k in range(len(error)):
                        for m in range(len(w[j])):
                            next_error = next_error + error[k] * w[j][m]

                    # Calculate weight change
                    if j == len(w) - 1:
                        for k in range(len(error)):
                            delta = self.learning_rate * np.sum(error[k] * z[j - 1])
                            w[j][k] = w[j][k] + delta
                    else:
                        for k in range(len(w[j])):
                            delta = self.learning_rate * z[j - 1] * np.sum(error[k] * z[j] * (1 - z[j]))
                            w[j][k] = w[j][k] + delta

                    error = next_error

                # Back propagate through first layer
                for j in range(len(w[0])):
                    delta = self.learning_rate * np.sum(error[j] * z[0] * (1 - z[0])) * _X[i]
                    w[0][j] = w[0][j] + delta

            self.w = w

            # Calculate predicted values after all weight updates
            predicted = np.empty((len(indices), len(self.classes)))
            for i in range(len(indices)):
                predicted[i] = self.calculate_output(_X[indices[i]], w)[-1]

            entropy = cross_entropy(_y, predicted)

            if entropy == 0:
                convergence_reached = True

            epochs = epochs + 1

    def fit_autoencoder(self, X):
        """
        Fit the autoencoder portion of the network
        :param X: A DataFrame of features
        :return: None
        """
        num_rows = X.shape[0]
        num_cols = X.shape[1]

        # Prepend bias input to each row
        _X = np.array(X)
        ones = np.ones((_X.shape[0], 1))
        _X_bias = np.concatenate((ones, _X), axis=1)

        # Initialize weight matrix
        self.w_a = np.empty(2, dtype=object)

        self.w_a[0] = np.random.uniform(-0.01, 0.01, (self.ae_layer, num_cols + 1))
        self.w_a[1] = np.random.uniform(-0.01, 0.01, (num_cols, self.ae_layer + 1))

        convergence_reached = False
        epochs = 0
        indices = list(range(num_rows))

        # Iterate through examples until convergence is reached
        while (not convergence_reached) and epochs < self.max_iter:
            # Shuffle data for stochastic gradient descent
            np.random.shuffle(indices)

            for i in indices:
                z = self.forward_propagate_autoencoder(_X_bias[i], self.w_a)

                # Back propagate
                error = _X[i] - z[1]

                delta_1 = 0
                for j in range(len(z[1])):
                    delta_1 = delta_1 + error * z[1][j]

                self.w_a[1] = self.w_a[1] + (self.learning_rate * np.sum(delta_1))

                delta_1 = np.sum(delta_1)
                delta_0 = 0
                for j in range(len(z[0])):
                    delta_0 = delta_0 + delta_1 * z[0][j] * (1 - z[0][j]) * _X_bias[i][j]

                self.w_a[0] = self.w_a[0] + (self.learning_rate * delta_0)

            entropy = 0
            for i in range(len(indices)):
                prediction = self.forward_propagate_autoencoder(_X_bias[i], self.w_a)[-1]
                entropy = cross_entropy(_X[i], prediction)

            entropy = -entropy

            if entropy == 0:
                convergence_reached = True

            epochs = epochs + 1

    def forward_propagate_autoencoder(self, x, w):
        """
        Calculate the forward propagation outputs given a data point and set of weights
        :param x: A data point
        :param w: A weight matrix
        :return: An output matrix
        """
        z = np.empty(len(w), dtype=object)

        for i in range(len(w)):
            if i == 0:
                z[i] = sigmoid(np.tensordot(w[i].T, x, axes=([0], [0])))
                z[i] = np.concatenate((np.ones(1), z[i]), axis=0)
            elif i == len(w) - 1:
                z[i] = sigmoid(np.tensordot(w[i], z[i - 1], axes=1))
            else:
                z[i] = sigmoid(np.tensordot(w[i], z[i - 1], axes=1))
                z[i] = np.concatenate((np.ones(1), z[i]), axis=0)

        return z

    @staticmethod
    def calculate_output(x, w):
        """
        Calculate the output given a data point and set of weights
        :param x: A data point
        :param w: A weight matrix for the first layer
        :return: An output matrix
        """
        z = np.empty(3, dtype=object)

        # Calculate output into first hidden layer
        z[0] = sigmoid(np.tensordot(w[0].T, x, axes=([0], [0])))
        z[0] = np.concatenate((np.ones(1), z[0]), axis=0)

        # Iterate through each hidden layer
        for j in range(1, len(w) - 1):
            z[j] = sigmoid(np.tensordot(w[j], z[j - 1], axes=1))
            z[j] = np.concatenate((np.ones(1), z[j]), axis=0)

        # Calculate final output
        z[len(z) - 1] = softmax(np.tensordot(w[len(w) - 1], z[len(z) - 2], axes=1))

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

        predicted = []
        for i in range(len(_X)):
            predicted.append(self.classes[np.argmax(self.calculate_output(_X[i], self.w)[-1])])

        return predicted
