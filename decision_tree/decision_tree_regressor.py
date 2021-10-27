# Decision tree regressor implemented with the CART algorithm
import sys

import numpy as np

from node import DiscreteLeafNode, DiscreteInteriorNode, ContinuousInteriorNode, ContinuousLeafNode
from utils.cross_validation import k_fold
from utils.evaluate import mean_squared_error
from utils.preprocessing import drop_column


class DecisionTreeRegressor:
    def __init__(self, theta):
        self.theta = theta
        self.tree = None
        self.X = None

    def fit(self, X, y):
        """
        Fit the model to the data
        :param X: A DataFrame of feature values
        :param y: A DataFrame of target values
        :return: None
        """
        self.X = X
        self.tree = self.generate_tree(X, y, None)

    def generate_tree(self, X, y, parent_node):
        """
        Generate a tree
        :param X: A DataFrame containing feature values
        :param y: A DataFrame containing target values
        :param parent_node: The parent node of the current node
        :return: A decision tree
        """
        # Calculate mean square error and check if it's below the purity threshold
        mu = np.array(y).mean()
        predicted = [mu] * y.shape[0]
        error = mean_squared_error(predicted, list(y))

        # If threshold has been reached, return a lead node
        if error < self.theta or X.shape[0] == 1:
            return ContinuousLeafNode(list(y), parent_node)

        # Otherwise, find the next best split
        feature, midpoint = self.split_attribute(X, y)

        # Recursively create discrete branches
        if midpoint is None:
            classes = list(self.X[feature].unique())

            node = DiscreteInteriorNode(list(y), feature, classes, parent_node)

            for cls in classes:
                indices = np.argwhere(np.array(X[feature]) == cls).flatten()

                if len(indices) == 0:
                    node.add_child(cls, DiscreteLeafNode(list(y), parent_node))
                else:
                    subset_X = X.iloc[indices]
                    subset_X = drop_column(subset_X, feature)

                    subset_y = y.iloc[indices]

                    node.add_child(cls, self.generate_tree(subset_X, subset_y, node))

        # Recursively create numeric branches
        else:
            node = ContinuousInteriorNode(y, feature, midpoint, parent_node)

            left_indices = np.argwhere(np.array(X[feature]) < midpoint).flatten()
            right_indices = np.argwhere(np.array(X[feature]) >= midpoint).flatten()

            # Check for empty nodes
            if len(left_indices) == 0 or len(right_indices) == 0:
                return ContinuousLeafNode(list(y), parent_node)

            left_X = X.iloc[left_indices]
            left_y = y.iloc[left_indices]

            node.add_child('left', self.generate_tree(left_X, left_y, node))

            right_X = X.iloc[right_indices]
            right_y = y.iloc[right_indices]
            node.add_child('right', self.generate_tree(right_X, right_y, node))

        return node

    def split_attribute(self, X, y):
        """
        Split best attribute
        :param X: A DataFrame containing feature values
        :param y: A DataFrame containing target values
        :return: The best split
        """
        min_error = sys.maxsize
        best_f = None
        best_midpoint = None

        attributes = list(X)

        for attribute in attributes:
            a_type = X.dtypes[attribute]

            # If the attribute is numeric
            if a_type in ['int64', 'float64']:
                # Find candidate splits around the median
                sorted_X = np.argsort(np.array(X[attribute]))
                med = int(len(sorted_X) / 2)
                percentage = int(len(sorted_X) * 0.05)

                candidates = list(range(med - percentage, med + percentage + 1))
                if len(candidates) <= 10:
                    candidates = sorted_X

                # Loop through all possible binary splits
                for index in range(1, len(candidates)):
                    # Find midpoint
                    midpoint = (X.iloc[candidates[index - 1]][attribute] + X.iloc[candidates[index]][attribute]) / 2

                    # Split data into two branches
                    left_indices = np.argwhere(np.array(X[attribute]) <= midpoint).flatten()
                    left = y.iloc[left_indices]

                    right_indices = np.argwhere(np.array(X[attribute]) > midpoint).flatten()
                    right = y.iloc[right_indices]

                    # Calculate mean square error after split
                    if left.shape[0] == 1:
                        error_left = 0
                    else:
                        predicted_left = np.array([left.mean()] * left.shape[0])
                        error_left = np.sum(np.square(np.subtract(np.array(left), predicted_left)))

                    if right.shape[0] == 1:
                        error_right = 0
                    else:
                        predicted_right = np.array([right.mean()] * right.shape[0])
                        error_right = np.sum(np.square(np.subtract(np.array(right), predicted_right)))

                    e = (error_left + error_right) / y.shape[0]

                    # Check if mean square error is less than previous best
                    if e < min_error:
                        min_error = e
                        best_f = attribute
                        best_midpoint = midpoint

            # If the attribute is discrete
            else:
                values = np.unique(X[attribute])

                # Calculate mean square error after split
                e = 0
                for value in values:
                    indices = np.argwhere(np.array(X[attribute]) == value).flatten()
                    subset = y.iloc[indices]

                    predicted = [subset.mean()] * subset.shape[0]
                    error = np.sum(np.square(np.subtract(np.array(list(subset)), predicted)))

                    e = e + error

                e = e / y.shape[0]

                # Check if mean square error is less than previous best
                if e < min_error:
                    min_error = e
                    best_f = attribute
                    best_midpoint = None

        return best_f, best_midpoint

    def predict(self, X):
        """
        Predict class of the given values
        :param X: A DataFrame of values
        :return: A list of predicted outputs
        """
        predictions = []

        # For each row in the data set, traverse the trained tree
        for index in range(X.shape[0]):
            node = self.tree
            node_type = node.get_type()

            row = X.iloc[index]

            while node_type == 'node':
                node = node.test(row)
                node_type = node.get_type()

            predictions.append(node.value)

        return predictions

    def show_tree(self):
        """
        Visualize a trained tree
        :return: None
        """
        self.draw_branch(self.tree, 0)

    def draw_branch(self, node, level_no):
        """
        Draw a branch of a tree
        :param node: The node to draw
        :param level_no: The depth of the node
        :return: None
        """
        branch = '|---'
        level = '|   '

        str = ''.join([level] * level_no)
        str = str + branch

        if node.get_type() == 'node':
            for child in node.children:
                if child == 'left':
                    print(f"{str}{node.feature} < {node.value}")
                elif child == 'right':
                    print(f"{str}{node.feature} >= {node.value}")
                else:
                    print(f"{str}{node.feature} == {child}")
                self.draw_branch(node.children[child], level_no + 1)
        else:
            print(f"{str}value: {node.value}")


def find_best_theta(data, target, theta_values):
    """
    Find the best theta (early stopping value)
    :param data: A DataFrame
    :param target: The target variable
    :param theta_values: Theta values to test
    :return: The best theta value
    """
    min_error = sys.maxsize
    best_theta = None

    # For each theta value perform 5-fold cross-validation
    for theta in theta_values:
        folds = k_fold(data, 5, validation=False)

        errors = []
        for fold in folds:
            train = fold['train']
            test = fold['test']

            train_X = drop_column(train, target)
            train_y = train[target]

            regressor = DecisionTreeRegressor(theta)
            regressor.fit(train_X, train_y)

            predicted = regressor.predict(drop_column(test, target))
            error = mean_squared_error(predicted, test[target])

            errors.append(error)

        avg_error = np.mean(errors)

        # Save theta value if less than value currently stored
        if avg_error < min_error:
            min_error = avg_error
            best_theta = theta

    return best_theta
