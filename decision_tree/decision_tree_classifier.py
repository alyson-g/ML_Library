# Decision tree classifier implemented with the ID3 algorithm
import numpy as np

from helpers import entropy, intrinsic_value
from node import DiscreteLeafNode, DiscreteInteriorNode, ContinuousInteriorNode
from utils.evaluate import classification_score
from utils.preprocessing import drop_column


class DecisionTreeClassifier:
    def __init__(self):
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
        :param parent_node: The parent node of the current tree
        :return: A decision tree
        """
        # If there are no more attributes or all examples are in same class, create a leaf node
        if len(list(X)) == 0 or len(y.unique()) == 1:
            return DiscreteLeafNode(list(y), parent_node)

        # Otherwise, find the next best split
        feature, midpoint = self.split_attribute(X, y)

        # Recursively create discrete branches
        if midpoint is None:
            classes = list(self.X[feature].unique())

            node = DiscreteInteriorNode(y, feature, classes, parent_node)
            for cls in classes:
                indices = np.argwhere(np.array(X[feature]) == cls).flatten()

                if len(indices) == 0:
                    node.add_child(cls, DiscreteLeafNode(list(y), node))
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
                return DiscreteLeafNode(list(y), node)

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
        # Calculate entropy before the split
        pre_split_ent = entropy(y)

        # Information needed to calculate intrinsic value
        total = X.shape[0]

        best_gain_ratio = 0
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
                    midpoint = (candidates[index - 1] + candidates[index]) / 2

                    # Split data into two branches
                    left = np.argwhere(np.array(X[attribute]) <= midpoint).flatten()
                    right = np.argwhere(np.array(X[attribute]) > midpoint).flatten()

                    # Calculate information gain of split
                    ent_left = (len(left) / total) * entropy(list(y.iloc[left]))
                    ent_right = (len(right) / total) * entropy(list(y.iloc[right]))

                    post_split_ent = -(ent_left + ent_right)

                    # Check if entropy is less than previous best
                    information_gain = pre_split_ent - post_split_ent
                    int_value = intrinsic_value([y.iloc[left], y.iloc[right]], total)
                    gain_ratio = information_gain / int_value

                    # Check if entropy is less than previous best
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_f = attribute
                        best_midpoint = None

            # If the attribute is discrete
            else:
                values = np.unique(X[attribute])

                # Calculate entropy after split
                post_split_ent = 0
                subsets_y = []

                for value in values:
                    indices = np.argwhere(np.array(X[attribute]) == value).flatten()
                    subset = X.iloc[indices]

                    subsets_y.append(y.iloc[indices])

                    sub_total = subset.shape[0]
                    ent = entropy(y.iloc[indices].tolist())
                    post_split_ent = post_split_ent + (sub_total / total) * ent

                post_split_ent = -post_split_ent

                # Check if entropy is less than previous best
                information_gain = pre_split_ent - post_split_ent
                int_value = intrinsic_value(subsets_y, total)
                gain_ratio = information_gain / int_value

                # Check if entropy is less than previous best
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
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

    def reduced_error_prune(self, X, y):
        """
        Perform reduced error pruning
        :param X: Pruning set features
        :param y: Pruning set classes
        :return: A pruned tree
        """
        # Calculate initial classification score
        predicted = self.predict(X)
        best_score = classification_score(predicted, list(y))

        # Indicator value to check if pruning has improved best score
        change_detected = True
        best_tree = self.tree

        # Continue to prune tree until no more improvement is possible
        while change_detected:
            pruned_tree = self.prune(best_tree)
            predictions = []

            for index in range(X.shape[0]):
                node = pruned_tree
                node_type = node.get_type()

                row = X.iloc[index]
                while node_type == 'node':
                    node = node.test(row)
                    node_type = node.get_type()

                predictions.append(node.value)

            score = classification_score(predictions, list(y))

            # If classification score has been improved, save the pruned tree
            if score > best_score:
                best_score = score
                best_tree = pruned_tree
            else:
                change_detected = False

        self.tree = best_tree

    def prune(self, node):
        """
        Prune a node
        :param node: The node to prune
        :return: The pruned node
        """
        if node.get_type() == 'leaf':
            return node

        if self.can_be_pruned(node):
            return DiscreteLeafNode(node.y, node.parent)
        else:
            for child in node.children:
                new_child = self.prune(node.children[child])
                node.children[child] = new_child

        return node

    def can_be_pruned(self, node):
        """
        Determine if a node can be pruned
        :param node: A node
        :return: Boolean value indicating if the node can be pruned
        """
        ret = True

        for child in node.children:
            ret = ret and node.children[child].get_type() == 'leaf'

        return ret
