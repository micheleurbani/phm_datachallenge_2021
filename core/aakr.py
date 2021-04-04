
import numpy as np
from sklearn.covariance import (
    empirical_covariance,
)
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold


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
        self.pipe = None  # placeholder for the transformation pipeline

    def fit(self, X, Y=None):
        """
        Apply fit operations and save the pipeline.

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like for compliance in pipeline use.

        Returns
        -------
        self
            The fitted estimator.

        """
        # Create pipeline for pre-processing of data
        self.pipe = make_pipeline(
            SimpleImputer(),
            VarianceThreshold(threshold=0.01),
        )
        # Apply the transformation
        self.pipe.fit(X)

    def transform(self, X, Y=None):
        """
        Apply the tranformation to the training `X` data and optionally to
        test `Y` data.

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Return
        ------
        X : (N_a, N_selected_features)
            array_like containing the transformed training dataset.
        Y : (N_b, N_selected_features)
            array_like containing the trasformed test dataset.
        """
        if self.pipe:
            X = self.pipe.transform(X)
            if Y is not None:
                Y = self.pipe.transform(Y)
                return X, Y
            else:
                return X
        else:
            raise BaseException("There is no fitted pipeline.")

    def fit_transform(self, X, Y):
        """
        Combines fit and transform method.

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Return
        ------
        X : (N_a, N_selected_features)
            array_like containing the transformed training dataset.
        Y : (N_b, N_selected_features)
            array_like containing the trasformed test dataset.
        """

        self.fit(X, Y)
        X, Y = self.transform(X, Y)
        return X, Y


    def predict(self, X, Y):
        """
        Reconstruct the signal.

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Returns
        -------
        Y_hat
            The reconstructed signal.
        """
        assert X.ndim >= 2
        assert Y.ndim >= 2
        # Fit  and transform the training and test data
        X, Y = self.fit_transform(X, Y)
        # Estimate the empirical covariance of the dataset X
        V = empirical_covariance(X)
        self.VI = np.zeros_like(V)
        np.fill_diagonal(self.VI, 1/np.diag(V))
        # Compute the Mahalanobis distance
        r2 = cdist(X, Y, metric='mahalanobis', VI=self.VI)
        # Compute the kernel
        self.w = (1 / (2 * np.pi * self.h**2)**0.5) * \
            np.exp(- r2**2 / (2 * self.h**2))
        # Reconstruct the signal
        return np.stack(
            [
                np.array(
                    [np.average(X[:, j], axis=0, weights=self.w[:, i])
                        for j in range(X.shape[1])]
                )
                for i in range(Y.shape[0])
            ]
        )
