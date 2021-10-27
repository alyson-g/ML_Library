# Defines node classes used to construct decision trees
import numpy as np

from helpers import most_frequent_class


class DiscreteLeafNode:
    def __init__(self, values, parent):
        self.value = most_frequent_class(values)
        self.parent = parent

    def get_type(self):
        """
        Get the node type
        :return: The type of node
        """
        return 'leaf'


class ContinuousLeafNode:
    def __init__(self, values, parent):
        self.value = np.mean(values)
        self.parent = parent

    def get_type(self):
        """
        Get the node type
        :return: The type of node
        """
        return 'leaf'


class DiscreteInteriorNode:
    def __init__(self, y, feature, classes, parent):
        self.y = y
        self.feature = feature
        self.parent = parent

        self.children = {}
        for cls in classes:
            self.children[cls] = None

    def test(self, data):
        """
        Evaluate a data point
        :param data: The data to test
        :return: The next node
        """
        return self.children[data[self.feature]]

    def add_child(self, cls, node):
        """
        Add a child node
        :param cls: The class value
        :param node: The child node
        :return: The correct child node
        """
        self.children[cls] = node

    def get_type(self):
        """
        Get the node type
        :return: The type of node
        """
        return 'node'


class ContinuousInteriorNode:
    def __init__(self, y, feature, value, parent):
        self.y = y
        self.feature = feature
        self.value = value
        self.children = {
            'left': None,
            'right': None
        }
        self.parent = parent

    def add_child(self, key, node):
        """
        Add a child node
        :param key: The key value to refer to the node
        :param node: The node to add
        :return: None
        """
        self.children[key] = node

    def test(self, data):
        """
        Evaluate a data point
        :param data
        :return: The correct child node
        """
        if data[self.feature] < self.value:
            return self.children['left']
        else:
            return self.children['right']

    def get_type(self):
        """
        Get the node type
        :return: The type of node
        """
        return 'node'
