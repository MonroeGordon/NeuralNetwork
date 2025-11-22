import cupy as cp
import numpy as np

class PCA:
    '''
    Principal Component Analysis class. Provides algorithms for performing principal component analysis for use in
    dimensionality reduction and data reconstruction.
    '''

    @staticmethod
    def covariance_matrix(x: np.ndarray | cp.ndarray,
                          device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Calculate the covariance matrix of a dataset.
        :param x: Input data matrix (number samples, number features).
        :param device: CPU or GPU device.
        :return: Covariance matrix.
        '''
        if x.ndim != 2:
            raise ValueError("PCA @ covariance_matrix: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            mu = np.mean(nx, axis=0)
            x_centered = nx - mu

            return np.dot(x_centered.T, x_centered) / (nx.shape[0] - 1)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = np.asarray(cx)

            mu = cp.mean(cx, axis=0)
            x_centered = cx - mu

            return np.dot(x_centered.T, x_centered) / (cx.shape[0] - 1)