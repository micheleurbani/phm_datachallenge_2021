
import numpy as np


class AAKR(object):
    """
    A wrapper class containing all the functions required to perform
    reconstruction of a signal using AAKR.

    Parameters:
    -----------

    training_data : a :class:`numpy.ndarray` containing the training data.

    """

    def __init__(self, training_data):
        self.X = training_data
