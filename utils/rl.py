# Functions used in reinforcement learning
import numpy as np
from scipy.stats import bernoulli


def read_track(file_name):
    """
    Read a track
    :param file_name: The name of the track file
    :return: A numpy array representation of a track
    """
    with open(f"./tracks/{file_name}") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    size = [int(s) for s in lines[0].split(',')]
    track = np.empty(tuple(size), dtype=str)

    for index in range(1, len(lines)):
        line = lines[index]

        for sub_index in range(len(line)):
            track[index - 1][sub_index] = line[sub_index]

    return track


def bernoulli_trial(probability):
    """
    Perform a Bernoulli trial
    :param probability: The success probability (between 0 and 1)
    :return:
    """
    return bool(bernoulli.rvs(probability, size=1)[0])
