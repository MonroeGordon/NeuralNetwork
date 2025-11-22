from algorithms.pca import PCA
from algorithms.lda import LDA
from linalg.decomposition import eigdecomp
from linalg.decomposition import svd_decomp
from linalg.eigen import eigen

import cupy as cp
import numpy as np

def pca_dim_reduction(x: np.ndarray | cp.ndarray,
                      n_components: int,
                      device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Reduce the dimensions of the dataset using principal component analysis (PCA).
    :param x: Input data matrix (number samples, number features).
    :param n_components: Number of principal components to keep.
    :param device: CPU or GPU device.
    :return: Transformed data.
    '''
    if x.ndim != 2:
        raise ValueError("@ pca_dim_reduction: parameter 'x' must be a matrix (2-dimensional ndarray).")

    c_mtx = PCA.covariance_matrix(x, device)
    e_vals, e_vecs = eigdecomp(c_mtx, device)

    w = e_vecs[:, :n_components]

    if device == "cpu":
        nx = x

        if isinstance(nx, cp.ndarray):
            nx = cp.asnumpy(nx)

        return np.dot(nx - np.mean(nx, axis=0), w)
    else:
        cx = x

        if isinstance(cx, np.ndarray):
            cx = cp.asarray(cx)

        return cp.dot(cx - cp.mean(cx, axis=0), w)

def lda_projection(x: np.ndarray | cp.ndarray,
                   y: list,
                   n_components: int,
                   device: str = "cpu") -> tuple:
    '''
    Perform linear discriminant analysis projection.
    :param x: Input feature matrix (number samples, number features).
    :param y: Class labels (number samples).
    :param n_components: Desired number of dimensions.
    :param device: CPU or GPU device.
    :return: Transformed feature matrix and the projection matrix.
    '''
    if x.ndim != 2:
        raise ValueError("@ lda_projection: parameter 'x' must be a matrix (2-dimensional ndarray).")

    means = LDA.means(x, y, device)
    sw = LDA.within_class_scatter(x, y, means, device)
    sb = LDA.between_class_scatter(x, y, means, device)

    if device == "cpu":
        e_vals, e_vecs = eigen(np.linalg.inv(sw).dot(sb))

        sorted_indices = np.argsort(e_vals)[::-1]
        e_vecs = e_vecs[:, sorted_indices]

        w = e_vecs[:, :n_components]
    else:
        e_vals, e_vecs = eigen(cp.linalg.inv(sw).dot(sb))

        sorted_indices = cp.argsort(e_vals)[::-1]
        e_vecs = e_vecs[:, sorted_indices]

        w = e_vecs[:, :n_components]

    return x.dot(w), w

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
    u, s, vt = svd_decomp(matrix, device)

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