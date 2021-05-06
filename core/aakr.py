
import numpy as np
import pandas as pd
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
        self.w = None  # placeholder for the weights of each observation in X
        self.VI = None  # placeholder for the inverse covariance matrix
        self.pipe = None  # placeholder for the transformation pipeline
        self.features = None  # placeholder for the selected featueres

    def __str__(self):
        return "<core.AAKR> \t h={:n}".format(self.h)

    def gaussian_rbf(self, distance):
        """
        Compute the Gaussian radial basis function.

        Parameters
        ----------
        distance : (N_a, N_b)
            array_like containing the distances between the observations and
            the training data. `N_a` corresponds to the number of samples in
            the training set, whereas `N_b` is the number of observations in
            the test set.
        Return
        ------
        self
            The object updated with the kernel weights.
        w : (N_a, N_b)
            array_like containing the weights of each training sample with
            respect to each observation.
        """
        w = (1 / (2 * np.pi * self.h**2)**0.5) * \
            np.exp(- distance**2 / (2 * self.h**2))
        self.w = w
        return w

    @staticmethod
    def mahalanobis_distance(a1, a2, VI):
        """
        Return the Mahalanobis distance between `a1` and `a2` using the
        inverse covariance matrix `VI`.

        Parameters
        ----------
        a1 : (N_a, N_features)
            array_like
        a2 : (N_b, N_features)
            array_like
        VI : (N_features, N_features)
            The diagonal matrix of the inverse of the covariance of the j-th
            hostorical signal (i.e., it is calculated using only the training
            data).

        Returns
        -------
        distance : (N_a, N_b)
            A matrix containing the Mahalanobis distance of the observations
            against the training data.
        """
        print(a1.shape, a2.shape)
        distance = np.zeros((a1.shape[0], a2.shape[0]))
        for j in range(a2.shape[0]):
            for i in range(a1.shape[0]):
                # Compute the difference and add a dimension
                distance[i, j] = np.dot(
                    np.dot(a2[j, :] - a1[i, :], VI),
                    a2[j, :] - a1[i, :]
                )
        return distance

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
        # Perform sanity checks
        assert type(X) is pd.DataFrame
        if Y is None:
            data = X
        else:
            assert type(Y) is pd.DataFrame
            data = pd.concat([X, Y])
        # Apply the transformation
        self.pipe.fit(
            data
        )
        # Store the names of the selected features
        mask = self.pipe.named_steps["variancethreshold"].get_support()
        self.features = X.columns[mask]

    def transform(self, X, Y=None):
        """
        Apply the tranformation to the training `X` data and test `Y` data.
        The invers covariance matrix is estimated and stored for future use.

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
            # Now that the data has been transformed, the inverse of the
            # covariance can be estimated and stored
            V = empirical_covariance(X)
            self.VI = np.zeros_like(V)
            np.fill_diagonal(self.VI, 1/np.diag(V))
            if Y is not None:
                Y = self.pipe.transform(Y)
                return X, Y
            else:
                return X, None
        else:
            raise BaseException("There is no fitted pipeline.")

    def fit_transform(self, X, Y=None):
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
        Reconstruct the signal. The method operates on numpy arrays internally,
        but it returns two pandas dataframes.

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Returns
        -------
        Y : (N_b, N_selected_features)
            array_like containing only the selected features of the observed
            signal.
        Y_hat : (N_b, N_selected_features)
            array_like containing only the selected features of the
            reconstructed signal.
        """
        assert X.ndim >= 2
        assert Y.ndim >= 2
        # If the pipe is empty, raise an error
        if self.pipe is None:
            raise BaseException("The estimator has not yet been fitted.")
        # Compute the Mahalanobis distance
        r2 = cdist(X, Y, metric='mahalanobis', VI=self.VI)
        # r2 = self.mahalanobis_distance(X, Y, self.VI)
        # Compute the kernel
        w = self.gaussian_rbf(distance=r2)
        # Reconstruct the signal
        x_nc = []
        for i in range(Y.shape[0]):
            x = []
            for j in range(X.shape[1]):
                x.append(np.average(X[:, j], weights=w[:, i]))
            x_nc.append(np.array(x))
        Y_hat = np.stack(x_nc)
        # Reconstruct the dataframes
        Y = pd.DataFrame(Y, columns=self.features)
        Y_hat = pd.DataFrame(Y_hat, columns=self.features)
        return Y, Y_hat

    def score(self, x_obs, x_nc):
        """
        Score the performance of the classifier using the Mean Square Error
        (MSE) of the reconstructions on the validation set.

        Parameters
        ----------
        x_obs : (N_a, N_features)
            array_like containing the observation data.
        x_nc : (N_a, N_features)
            array_like containing the reconstructed signals.

        Returns
        -------
        score : scalar
            The MSE value obtained on the given training dataset.
        """
        n_samples = x_obs.shape[0]
        return np.sum((x_obs - x_nc)**2 / n_samples).sum()

    def _reset(self):
        """
        Reset the attibutes of the estimator.
        """
        self.VI = None
        self.features = None
        self.pipe = None
        self.w = None


class ModifiedAAKR(AAKR):
    """
    Implements the modified AAKR method in [BARALDI2015_].

    .. [BARALDI2015] Piero Baraldi, Francesco Di Maio, Pietro Turati,
    Enrico Zio, Robust signal reconstruction for condition monitoring of
    industrial components via a modified Auto Associative Kernel Regression
    method, *Mechanical Systems and Signal Processing* 60–61:29-44 (2015)
    https://doi.org/10.1016/j.ymssp.2014.09.013
    """

    def __init__(self, h=1.0, p=None, k=2.0):
        super().__init__(h=h)
        self.p = p  # placeholder for the vector of penalties
        self.k = k

        self.D = None  # Placeholder for the diagonal matrix with penalties

    def __str__(self):
        return "<core.AAKR> \t h={:n}".format(self.h)

    @staticmethod
    def abs_normalized_distance(self, X, Y):
        """
        Implements the computation of the distance between the observations and
        the training data as in [BARALDI2015_].

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
            array_like is the matrix which, when multiplied to a vector,
            orders the vector components in decreasing order.
        """
        n_features = len(x_obs)
        sort_indexes = np.flip(np.argsort(x_obs))
        P = np.zeros((n_features, n_features))
        for i, _ in enumerate(P):
            P[sort_indexes[i], i] = 1
        return P

    @staticmethod
    def transformation(D, P, x_obs):
        """
        Implement the non-homogeneous transformation introduced in
        [BARALDI2015_].

        Parameters
        ----------
        D : (N_features, N_features)
            array_like, it is the diagonal matrix having increasing entries so
            that :math:`tr(\\boldsymbol{D}) = \\boldsymbol{p}`.

        P : (N_features, N_features)
            array_like, the permutation matrix obtained through
            :method:`core.aakr.ModifiedAAKR.permutation_matrix`.

        x_obs : (N_features, )
            array_like is the observation vector.

        Return
        ------
        psi : (N_features, )
            array_like, it is the projection of `x_obs` in a new space.
        """
        return np.dot(np.dot(D, P), x_obs)

    def projection(self, X, Y, abs_norm_dist):
        """
        Perform the projection of a trajectory into the new space, as in
        [BARALDI2015_].

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing the set of training observations.
        Y : (N_b, N_features)
            array_like containing the observed signals.

        Returns
        -------
        phi_X : 

        phi_Y : 
        """
        abs_norm_dist = self.abs_normalized_distance(X, Y)
        # Compute the absolute normalized differences
        for i, y in enumerate(Y):
            for j, x in enumerate(X):
                # Compute the permuation matrix
                self.permutation_matrix(abs_norm_dist[i, j, :])
                # Project the training observation and the observed signal
                self.transformation()

    def fit(self):
        """
        Add the penalty vector of the right size defined according to the
        attribute `penalty_mode`, and build the matrix `D`.
        """
        super.fit()
        # Define a penalty vector of the right size.
        self.p = np.array([self.k**(2*(i+1))
                          for i in range(len(self.features))])
        # Define the diagonal matrix `D`
        self.D = np.zeros_like(self.p)
        np.fill_diagonal(self.D, np.sqrt(self.p))

    def predict(self, X, Y):
        """
        Reconstruct the signal according to the modified AAKR proposed by
        [BARALDI2015_].

        Parameters
        ----------
        X : (N_a, N_features)
            array_like containing training data.
        Y : (N_b, N_features)
            array_like containing the observations to be reconstructed.

        Returns
        -------
        Y : (N_b, N_selected_features)
            array_like containing only the selected features of the observed
            signal.
        Y_hat : (N_b, N_selected_features)
            array_like containing only the selected features of the
            reconstructed signal.
        """
        assert X.ndim >= 2
        assert Y.ndim >= 2
        # If the pipe is empty, raise an error
        if self.pipe is None:
            raise BaseException("The estimator has not yet been fitted.")
        # Compute the projection of X (x_obs_nc) and Y (x_obs)
        phi_X, phi_Y = self.projection(X, Y)
        # Compute the Euclidean distance between the points
        r2 = cdist(phi_X, phi_Y, metric='euclidean')
        # Compute the kernel
        w = self.gaussian_rbf(distance=r2)
        # Reconstruct the signal
        x_nc = []
        for i in range(Y.shape[0]):
            x = []
            for j in range(X.shape[1]):
                x.append(np.average(X[:, j], weights=w[:, i]))
            x_nc.append(np.array(x))
        Y_hat = np.stack(x_nc)
        # Reconstruct the dataframes
        Y = pd.DataFrame(Y, columns=self.features)
        Y_hat = pd.DataFrame(Y_hat, columns=self.features)
        return Y, Y_hat
