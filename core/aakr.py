
import numpy as np


class AAKR(object):
    """
    A wrapper class containing all the functions required to perform
    reconstruction of a signal using AAKR.

    Parameters:
    -----------

    training_data : a :class:`pandas.DataFrame` containing the training data.

    """

    def __init__(self, training_data):
        self.X = training_data

    def variance_matrix(self):
        cov = self.X.cov().to_numpy().diagonal()
        variance = np.fill_diagonal(np.zeros_like(cov), cov.diagonal())
        return variance
