import cupy as cp
import numpy as np

class Loss:
    @staticmethod
    def bin_cross_entropy(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the binary cross entropy loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Binary cross entropy loss.
        '''
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p), axis=1)

    @staticmethod
    def bin_cross_entropy_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the binary cross entropy loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Binary cross entropy loss.
        '''
        return -cp.mean(y * cp.log(p) + (1 - y) * cp.log(1 - p), axis=1)

    @staticmethod
    def cat_cross_entropy(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the categorical cross entropy loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Categorical cross entropy loss.
        '''
        return -np.sum(y * np.log(p), axis=1)

    @staticmethod
    def cat_cross_entropy_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the categorical cross entropy loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Categorical cross entropy loss.
        '''
        return -cp.sum(y * cp.log(p), axis=1)

    @staticmethod
    def hinge(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the hinge loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Hinge loss.
        '''
        return np.mean(np.maximum(0, 1 - y * p), axis=1)

    @staticmethod
    def hinge_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the hinge loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Hinge loss.
        '''
        return cp.mean(cp.maximum(0, 1 - y * p), axis=1)

    @staticmethod
    def huber(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the huber loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters. Uses huber_delta.
        :return: Huber loss.
        '''
        h = 1.0

        if hyper_param is not None and "huber_delta" in hyper_param.keys():
            h = hyper_param["huber_delta"]

        r = np.abs(y - p)
        q_region = r < h
        loss_q = 0.5 * (r[q_region])**2
        loss_l = h * (r[~q_region] - 0.5 * h)
        total_loss = np.sum(loss_q, axis=1) + np.sum(loss_l, axis=1)

        return total_loss / y.shape[1]

    @staticmethod
    def huber_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict = None) -> cp.ndarray:
        '''
        Compute the huber loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters. Uses huber_delta.
        :return: Huber loss.
        '''
        h = 1.0

        if hyper_param is not None and "huber_delta" in hyper_param.keys():
            h = hyper_param["huber_delta"]

        r = cp.abs(y - p)
        q_region = r < h
        loss_q = 0.5 * (r[q_region]) ** 2
        loss_l = h * (r[~q_region] - 0.5 * h)
        total_loss = cp.sum(loss_q, axis=1) + cp.sum(loss_l, axis=1)

        return total_loss / y.shape[1]

    @staticmethod
    def kl_divergence(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the Kullback-Leibler (KL) divergence loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Kullback-Leibler (KL) divergence loss.
        '''
        epsilon = 1e-10
        q = np.maximum(p, epsilon)

        return np.sum(y * np.log(y / q), axis=1)

    @staticmethod
    def kl_divergence_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict = None) -> cp.ndarray:
        '''
        Compute the Kullback-Leibler (KL) divergence loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Kullback-Leibler (KL) divergence loss.
        '''
        epsilon = 1e-10
        q = cp.maximum(p, epsilon)

        return cp.sum(y * cp.log(y / q), axis=1)

    @staticmethod
    def l1_mae(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the mean absolute error (L1) loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Mean absolute error (L1) loss.
        '''
        return np.mean(np.abs(y - p), axis=1)

    @staticmethod
    def l1_mae_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the mean absolute error (L1) loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Mean absolute error (L1) loss.
        '''
        return cp.mean(cp.abs(y - p), axis=1)

    @staticmethod
    def l2_mse(y: np.ndarray, p: np.ndarray, hyper_param: dict = None) -> np.ndarray:
        '''
        Compute the mean squared error (L2) loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Mean squared error (L2) loss.
        '''
        return np.mean(np.square(y - p), axis=1)

    @staticmethod
    def l2_mse_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict = None) -> cp.ndarray:
        '''
        Compute the mean squared error (L2) loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Mean squared error (L2) loss.
        '''
        return cp.mean(cp.square(y - p), axis=1)

    @staticmethod
    def log_cosh(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the log-cosh loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Log-cosh loss.
        '''
        error = p - y
        return np.mean(np.log(np.exp(error) + np.exp(-error)) - np.log(2), axis=1)

    @staticmethod
    def log_cosh_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Compute the log-cosh loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Log-cosh loss.
        '''
        error = p - y
        return cp.mean(cp.log(cp.exp(error) + cp.exp(-error)) - cp.log(2), axis=1)

    @staticmethod
    def sparse_cat_cross_entropy(y: np.ndarray, p: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Compute the sparse categorical cross entropy loss function.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Sparse categorical cross entropy loss.
        '''
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1.0 - epsilon)
        correct = p[np.arange(1), y]

        return -np.sum(np.log(correct), axis=1)

    @staticmethod
    def sparse_cat_cross_entropy_gpu(y: cp.ndarray, p: cp.ndarray, hyper_param: dict = None) -> cp.ndarray:
        '''
        Compute the sparse categorical cross entropy loss function on the GPU.
        :param y: True value(s).
        :param p: Predicted value(s).
        :param hyper_param: Hyperparameters.
        :return: Sparse categorical cross entropy loss.
        '''
        epsilon = 1e-10
        p = cp.clip(p, epsilon, 1.0 - epsilon)
        correct = p[cp.arange(1), y]

        return -cp.sum(cp.log(correct), axis=1)