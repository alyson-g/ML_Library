# Data pre-processing functions
import numpy as np
import pandas as pd
import pandas.api.types as types


def log_transform(data, column):
    """
    Log transform the given column
    :param data: A DataFrame
    :param column: The column name to transform
    :return: The transformed column
    """
    return data[column].apply(np.log)


def drop_column(data, column):
    """
    Drop a column from the dataset
    :param data: A DataFrame
    :param column: The column to remove
    :return: The transformed DataFrame
    """
    return data.drop(column, axis=1)


def rename_column(data, old_column, new_column):
    """
    Rename the given column
    :param data: A DataFrame
    :param old_column: The old column name
    :param new_column: The new column name
    :return: The transformed DataFrame
    """
    data[new_column] = data[old_column]
    data = drop_column(data, old_column)
    return data


def substitute_value(data, column, old_value, new_value):
    """
    Substitute one value in a particular column for another
    :param data: A DataFrame
    :param column: The column to transform
    :param old_value: The old value to substitute
    :param new_value: The new value to substitute with
    :return: The transformed column
    """
    return data[column].apply(lambda x: new_value if x == old_value else x)


def impute_mean(data, column, missing_val):
    """
    Impute missing values with the mean
    :param data: A DataFrame
    :param column: The column to transform
    :param missing_val: The expected missing value
    :return: The transformed column
    """
    data[column] = data[column].replace(missing_val, np.nan)

    # Ensure data is of numeric type before filling in mean
    data[column] = pd.to_numeric(data[column])

    data[column].fillna(value=data[column].mean(), inplace=True)
    return data[column]


def encode_ordinal_as_int(data, column, ordering):
    """
    Encode ordinal data as integers
    :param data: A DataFrame
    :param column: The column to transform
    :param ordering: An array of strings indicating the ordering of the values
    :return: The transformed column
    """
    # Create a mapping of the ordering to integers
    mapping = {}
    for index in range(len(ordering)):
        mapping[ordering[index]] = index

    # Apply mapping to the specified column
    return data[column].apply(lambda x: mapping[x])


def one_hot_encode(data, column, keep_column=False):
    """
    One-hot encode the given column
    :param data: A DataFrame
    :param column: The column to transform
    :param keep_column: Whether or not to keep the original column
    :return: The transformed DataFrame
    """
    dummies = pd.get_dummies(data[column], prefix=column)
    data = pd.concat([data, dummies], axis=1)
    data = data.reset_index(drop=True)

    if not keep_column:
        data = drop_column(data, column)

    return data


def discretize(data, column, discretize_type, bins):
    """
    Discretize a variable
    :param data: A DataFrame
    :param column: The column to be discretized
    :param discretize_type: Either 'equal_width' or 'equal_frequency'
    :param bins: The number of bins
    :return: The transformed column
    """
    if discretize_type not in ['equal_width', 'equal_frequency']:
        raise 'Invalid discretize_type inputted'

    if discretize_type == 'equal_width':
        return pd.cut(np.array(data[column]), bins)
    else:
        return pd.qcut(np.array(data[column]), bins)


def standardize(train, test):
    """
    Standardize all features in the train and test sets
    :param train: A DataFrame representing the train set
    :param test: A DataFrame representing the test set
    :return: The standardized train and test sets
    """
    for col in list(train):
        if types.is_numeric_dtype(train[col]):
            mu = train[col].mean()
            sigma = train[col].std()

            train[col] = train[col].apply(lambda x: (x - mu) / sigma)
            test[col] = test[col].apply(lambda x: (x - mu) / sigma)

    return train, test


def mean_scale(data, column):
    """
    Mean scale a column
    :param data: A DataFrame
    :param column: The column to scale
    :return: The scaled column
    """
    mu = np.mean(data[column])
    return data[column] / mu
