
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
        self.S = self.inverse_variance_matrix()

    def inverse_variance_matrix(self):
        cov = self.X.cov().to_numpy()
        inv_variance = np.zeros_like(cov)
        np.fill_diagonal(inv_variance, 1 / cov.diagonal())
        return inv_variance

    @staticmethod
    def mahalanobis_distance(u, v, S):
        """
        Compute the Mahalanobis distance between two vectors `u` and `v`.

        Parameters
        ----------
        u : (N, ) array_like
            Input array.

        v : (N, ) array_like
            Input array.

        S : (N, N) ndarray
            The inverse covariance matrix.

        Returns
        -------
        The square of the Mahalanobis distance between `u` and `v`.
        """
        S = np.atleast_2d(S)
        delta = u - v
        return np.dot(np.dot(delta, np.linalg.inv(S)), delta)

    def gaussian_radial_basis_function(self, u, v, h):
        """
        Compute the scalar value defined by the standard radial basis function
        with bandwidth parameter `h`.

        Parameters
        ----------
        u : (N, ) array_like
            Input array.

        v : (N, ) array_like
            Input array.

        h : scalar
            The bandwidth parameter.

        Returns
        -------
        The scalar value of the radial basis function.
        """
        return (2 * np.pi * h**2)**(-1) * \
            np.exp(-self.mahalanobis_distance(u, v, self.S) / 2 * h**2)
