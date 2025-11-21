from linalg.decomposition import svd

import cupy as cp
import numpy as np

def svd_dim_reduction(matrix: np.ndarray | cp.ndarray,
                      k: int,
                      device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Reduce the dimensionality of the input matrix using SVD.
    :param matrix: Input matrix (2-dimensional ndarray).
    :param k: Number of singular values to retain.
    :param device: CPU or GPU device.
    :return: Reduced matrix.
    '''
    u, s, vt = svd(matrix, device)

    if device == "cpu":
        uk = u[:, :k]
        sk = np.diag(s[:k])
        vtk = vt[:k, :]

        return np.dot(uk, np.dot(sk, vtk))
    else:
        uk = u[:, :k]
        sk = cp.diag(s[:k])
        vtk = vt[:k, :]

        return cp.dot(uk, cp.dot(sk, vtk))