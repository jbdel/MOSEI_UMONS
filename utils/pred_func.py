import numpy as np


def amax(x):
    return np.argmax(x, axis=1)


def multi_label(x):
    return (x > 0)