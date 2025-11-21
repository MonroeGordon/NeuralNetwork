from tools.reduction import svd_dim_reduction

import cupy as cp
import numpy as np

def svd_img_compression(matrix: np.ndarray | cp.ndarray,
                        k: int,
                        device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Compress an image matrix using SVD.
    :param matrix: Image matrix.
    :param k: Number of singular values to retain.
    :param device: CPU or GPU device.
    :return: Compressed image matrix.
    '''
    svd_dim_reduction(matrix, k, device)