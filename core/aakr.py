
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import (
    empirical_covariance,
)
from scipy.spatial.distance import cdist


class AAKR(object):
    """
    A wrapper class containing all the functions required to perform
    reconstruction of a signal using AAKR.

    Parameters
    ----------
    h : float
        The bandwidth parameter to compute the radial basis function.

    """

    def __init__(self, h=1.0):
        # Properties
        self.h = h

        # Attributes
        self.w = None  # the weights of each observation in X
        self.VI = None  # placeholder for the inverse covariance matrix

    def predict(self, X, Y):
        """

        Parameters:
        -----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Returns
        -------
        self
            Fitted estimator.
        """
        assert X.ndim >= 2
        assert Y.ndim >= 2
        # Estimate the empirical covariance of the dataset X
        V = empirical_covariance(X)
        self.VI = np.zeros_like(V)
        np.fill_diagonal(self.VI, 1/np.diag(V))
        # Compute the Mahalanobis distance
        r2 = cdist(X, Y, metric='mahalanobis', VI=self.VI)
        # Compute the kernel
        self.w = (1 / (2 * np.pi * self.h**2)**0.5) * np.exp(- r2**2 / (2 * self.h**2))
