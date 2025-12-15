from scipy.stats import multivariate_normal

import cupy as cp
import numpy as np

class GMM:
    '''
    Gaussian Mixture Model (GMM) class. Allows for creating a gaussian mixture model for tasks such as density
    estimation, clustering, generative modeling, or anomaly detection.
    '''

    def __init__(self,
                 n_components: int,
                 max_iter: int=100,
                 tolerance: float=1e-6,
                 device: str="cpu"):
        '''
        Initializes a gaussian mixture model.
        :param n_components: Number of gaussian components.
        :param max_iter: Maximum number of training iterations.
        :param tolerance: Tolerance value.
        :param device: CPU or GPU device.
        '''
        if n_components < 1:
            raise ValueError("GMM: parameter 'n_component' must exceed 0.")

        if max_iter < 1:
            raise ValueError("GMM: parameter 'max_iter' must exceed 0.")

        if not tolerance > 0.0:
            raise ValueError("GMM: parameter 'tolerance' must exceed 0.0.")

        if device != 'cpu' and device != 'gpu':
            raise ValueError("GMM: parameter 'device' must be 'cpu' or 'gpu'.")

        self._n_components = n_components
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._device = device
        self._weights = None
        self._means = None
        self._covariances = None

    def _e_step(self, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        '''
        Calculate the expected values of the latent variables.
        :param x: Input feature matrix (number samples, number features).
        :return: Expected latent variable values.
        '''
        if self._device == 'cpu':
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            likelihoods = np.array([self._weights[k] *
                                    multivariate_normal.pdf(nx, mean=self._means[k], cov=self._covariances[k])
                                    for k in range(self._n_components)])
            responsibilities = likelihoods / np.sum(likelihoods, axis=0)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            likelihoods = cp.array([self._weights[k] *
                                    multivariate_normal.pdf(cx, mean=self._means[k], cov=self._covariances[k])
                                    for k in range(self._n_components)])
            responsibilities = likelihoods / cp.sum(likelihoods, axis=0)

        return responsibilities.T

    def _log_likelihood(self, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        '''
        Compute the log likelihood of the input feature data.
        :param x: Input feature matrix (number samples, number features).
        :return: Log likelihood.
        '''
        if self._device == 'cpu':
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            return np.sum(np.log(np.sum([self._weights[k] *
                                         multivariate_normal.pdf(nx, mean=self._means[k], cov=self._covariances[k])
                                         for k in range(self._n_components)], axis=0)))
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = np.asarray(cx)

            return cp.sum(cp.log(cp.sum(
                cp.array([self._weights[k] * multivariate_normal.pdf(cx, mean=self._means[k], cov=self._covariances[k])
                          for k in range(self._n_components)]), axis=0)))

    def _m_step(self,
                x: np.ndarray | cp.ndarray,
                responsibilities: np.ndarray | cp.ndarray):
        '''
        Calculate the maximization of the likelihoods.
        :param x: Input feature matrix (number samples, number features).
        :param responsibilities: Expected latent variable values.
        '''
        if self._device == 'cpu':
            nx = x
            nr = responsibilities

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(nr, cp.ndarray):
                nr = cp.asnumpy(nr)

            resp_sum = nr.sum(axis=0)
            self._weights = resp_sum / nx.shape[0]
            self._means = np.array([np.sum(nr[:, k][:, np.newaxis] * nx, axis=0) / resp_sum[k]
                                    for k in range(self._n_components)])

            for k in range(self._n_components):
                diff = nx - self._means[k]
                self._covariances[k] = np.dot(nr[:, k] * diff.T, diff) / resp_sum[k]
        else:
            cx = x
            cr = responsibilities

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cr, np.ndarray):
                cr = cp.asarray(cr)

            resp_sum = cr.sum(axis=0)
            self._weights = resp_sum / cx.shape[0]
            self._means = cp.array([np.sum(cr[:, k][:, np.newaxis] * cx, axis=0) / resp_sum[k]
                                    for k in range(self._n_components)])

            for k in range(self._n_components):
                diff = cx - self._means[k]
                self._covariances[k] = cp.dot(cr[:, k] * diff.T, diff) / resp_sum[k]

    def fit(self, x: np.ndarray | cp.ndarray) -> list:
        '''
        Trains the gaussian mixture model on the input feature data.
        :param x: Input feature matrix (number samples, number features).
        :return: List of log-likelihoods.
        '''
        n_samples, n_features = x.shape
        log_likelihoods = []

        if self._device == 'cpu':
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            self._weights = np.ones(self._n_components) / self._n_components
            self._means = np.random.rand(self._n_components, n_features)
            self._covariances = np.array([np.eye(n_features) for _ in range(self._n_components)])

            for iteration in range(self._max_iter):
                responsibilities = self._e_step(nx)

                self._m_step(nx, responsibilities)

                log_likelihood = self._log_likelihood(nx)
                log_likelihoods.append(log_likelihood)

                if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < self._tolerance:
                    break
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            self._weights = cp.ones(self._n_components) / self._n_components
            self._means = cp.random.rand(self._n_components, n_features)
            self._covariances = cp.array([cp.eye(n_features) for _ in range(self._n_components)])

            for iteration in range(self._max_iter):
                responsibilities = self._e_step(cx)

                self._m_step(cx, responsibilities)

                log_likelihood = self._log_likelihood(cx)
                log_likelihoods.append(log_likelihood)

                if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < self._tolerance:
                    break

        return log_likelihoods

    def predict(self, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        '''
        Predict outputs using the gaussian mixture model based on the input feature data.
        :param x: Input feature matrix (number samples, number features).
        :return: Predicted output labels.
        '''
        if self._device == 'cpu':
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            return np.argmax(self._e_step(nx), axis=1)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            return cp.argmax(self._e_step(cx), axis=1)