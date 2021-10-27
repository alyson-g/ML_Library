# Cross-validation functions
import pandas as pd


def k_fold(data, k, validation=True):
    """
    Generate k folds for cross validation
    :param data: A DataFrame
    :param k: The number of folds to generate
    :param validation: Whether or not to generate a validation set
    :return: K folds in array form
    """
    # First randomly shuffle data
    data = data.sample(frac=1)

    # Split off validation set first, if required
    if validation:
        # Get number of values to include in validation set
        num_vals = int(0.20 * data.shape[0])

        val_set = data.iloc[:num_vals]
        data = data.iloc[num_vals:]

    # Next, split data into k folds
    folds = []

    # Get the size of each fold
    size = data.shape[0] / k

    for index in range(k):
        fold = {}

        # Calculate indices for train and test sets
        start_index = int(index * size)
        if index == k - 1:
            test = data.iloc[start_index:]
            train = data.iloc[:start_index]
        else:
            end_index = int(start_index + size)

            test = data.iloc[start_index:end_index]
            train = pd.concat([data.iloc[:start_index], data.iloc[end_index:]])

        fold['train'] = train
        fold['test'] = test

        folds.append(fold)

    if validation:
        return folds, val_set
    else:
        return folds


def sorted_k_fold(data, y, k, validation=True):
    """
    Generate k folds for cross validation
    :param data: A DataFrame
    :param y: The target variable
    :param k: The number of folds to generate
    :param validation: Whether or not to generate a validation set
    :return: K folds in array form
    """
    # Split off validation set first, if required
    if validation:
        # First randomly shuffle data
        data = data.sample(frac=1)

        # Get number of values to include in validation set
        num_vals = int(0.20 * data.shape[0])

        val_set = data.iloc[:num_vals]
        data = data.iloc[num_vals:]

    # First, sort data
    data = data.sort_values(y, axis=0)

    # Next, split data into 5 sets
    sets = []

    for index in range(k):
        sets.append([])

    total = data.shape[0]
    set_index = 0

    for index in range(total):
        sets[set_index].append(index)

        if set_index == k - 1:
            set_index = 0
        else:
            set_index = set_index + 1

    # Finally create training and test sets
    folds = []

    for index in range(k):
        fold = {}
        test = data.iloc[sets[index]]

        train = pd.DataFrame()
        for sub_index in range(k):
            if sub_index == index:
                continue

            train = pd.concat([train, data.iloc[sets[sub_index]]])

        fold['train'] = train
        fold['test'] = test

        folds.append(fold)

    if validation:
        return folds, val_set
    else:
        return folds


def stratified_k_fold(data, y, k, validation=True):
    """
    Generate stratified k folds for cross validation
    :param data: A DataFrame
    :param y: The target variable
    :param k: The number of folds to generate
    :param validation: Whether or not to generate a validation set
    :return:
    """
    # First randomly shuffle data
    data = data.sample(frac=1)

    # Find unique classes
    target_classes = list(data[y].unique())
    target_sets = {}

    for cls in target_classes:
        target_sets[cls] = data[data[y] == cls]

    # Split off validation set first, if required
    if validation:
        validation_set = pd.DataFrame()

        # Get number of values to include in validation set
        for cls in target_sets:
            size = int(0.20 * target_sets[cls].shape[0])
            sub_val_set = target_sets[cls].iloc[:size]

            target_sets[cls] = target_sets[cls].iloc[size:]

            validation_set = pd.concat([validation_set, sub_val_set])

    # Stratify train sets
    train_sets = {}
    for cls in target_sets:
        train_sets[cls] = []
        size = int(target_sets[cls].shape[0] / k)

        for index in range(k):
            start_index = index * size

            if index == k - 1:
                subset = target_sets[cls].iloc[start_index:]
            else:
                end_index = start_index + size
                subset = target_sets[cls].iloc[start_index:end_index]

            train_sets[cls].append(subset)

    # Create k folds
    folds = []

    for index in range(k):
        fold = {}

        train_set = pd.DataFrame()
        test_set = pd.DataFrame()

        for cls in target_sets:
            test_set = pd.concat([test_set, train_sets[cls][index]])

            for sub_index in range(k):
                if sub_index == index:
                    continue
                train_set = pd.concat([train_set, train_sets[cls][sub_index]])

        fold['train'] = train_set.sample(frac=1)
        fold['test'] = test_set.sample(frac=1)

        folds.append(fold)

    if validation:
        return folds, validation_set
    else:
        return folds
