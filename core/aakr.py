
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

    def gaussian_rbf(self, distance):
        """
        Compute the Gaussian radial basis function.

        Parameters
        ----------
        distance : (N_a)
            array_like containing the distances between the observation and the
            training data.
        Return
        ------
        self
            The objected updated with the kernel weights.
        """
        self.w = (1 / (2 * np.pi * self.h**2)**0.5) * \
            np.exp(- distance**2 / (2 * self.h**2))

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
        mask = self.pipe.named_steps["variancethreshold"].get_support()
        self.features = X.columns[mask]

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
        self.gaussian_rbf(distance=r2)
        # Reconstruct the signal
        Y_hat = np.stack(
            [
                np.array(
                    [np.average(X[:, j], axis=0, weights=self.w[:, i])
                        for j in range(X.shape[1])]
                )
                for i in range(Y.shape[0])
            ]
        )
        return Y, Y_hat


class ModifiedAAKR(AAKR):

    def __init__(self, h=1.0):
        super().__init__(h=h)

    def abs_normalized_distance(self, X, Y):
        """
        Implements the computation of the distance between the observations and
        the training data as in [BARALDI2015_]:

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Returns
        -------
        distance : (N_a, N_b, N_features)
            array_like containing the distances of the i-th observation from
            the j-th training example.

        .. [BARALDI2015] Piero Baraldi, Francesco Di Maio, Pietro Turati,
        Enrico Zio, Robust signal reconstruction for condition monitoring of
        industrial components via a modified Auto Associative Kernel Regression
        method, *Mechanical Systems and Signal Processing* 60–61:29-44 (2015)
        https://doi.org/10.1016/j.ymssp.2014.09.013
        """
        def dist(x, y, V):
            return np.abs(np.divide((x - y), V))
        V = np.var(X, axis=0)
        return np.stack([np.stack([dist(x, y, V) for x in X]) for y in Y])

    @staticmethod
    def permutation_matrix(x_obs):
        """
        Returns the matrix that order the elements of `x_obs` in decreasing
        order, i.e. from the largest to the smallest difference.

        Parameters
        ----------
        x_obs : (N_features, )
            array_like is the observation vector.

        Return
        ------
        permuation_matrix : (N_features, N_features)
            array_like is the matrix which, when multiplied to a vector, only
            modifies the order of the vector components.
        """
        n_features = len(x_obs)
        sort_indexes = np.flip(np.argsort(x_obs))
        P = np.zeros((n_features, n_features))
        for i, _ in enumerate(P):
            P[sort_indexes[i], i] = 1
        return P
