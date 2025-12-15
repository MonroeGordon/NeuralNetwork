from neural.loss import Loss

import cupy as cp
import numpy as np

class SVM:
    '''
    Support Vector Machine (SVM) class. Allows for creating a Support Vector Machine model that can create a linear
    hyperplane for data classification or regression.
    '''
    def __init__(self,
                 learning_rate: float=0.001,
                 kernel: str="linear",
                 hyper_param: dict=None,
                 device: str="cpu"):
        '''
        Initializes a support vector machine neural network model.
        :param learning_rate: Learning rate.
        :param kernel: Kernel trick for mapping input data to higher dimensions. May use 'linear', 'polynomial', 'rbf',
        or 'sigmoid'.
        :param hyper_param: Hyperparameters. May use 'svm_bias', 'svm_degree', or 'svm_gamma'.
        :param device: CPU or GPU device.
        '''
        if not learning_rate > 0.0:
            raise ValueError("SVM: parameter 'learning_rate' must exceed 0.0.")

        if device != "cpu" and device != "gpu":
            raise ValueError("SVM: parameter 'device' must be 'cpu' or 'gpu'.")

        self._weights = None
        self._biases = np.zeros((1, 1)) if device == "cpu" else cp.zeros((1, 1))
        self._learning_rate = learning_rate
        self._kernels = {
            "linear": {"cpu": SVM._linear,
                       "gpu": SVM._linear_gpu},
            "polynomial": {"cpu": SVM._polynomial,
                           "gpu": SVM._polynomial_gpu},
            "rbf": {"cpu": SVM._rbf,
                    "gpu": SVM._rbf_gpu},
            "sigmoid": {"cpu": SVM._sigmoid,
                        "gpu": SVM._sigmoid_gpu},
        }
        self._kernel = kernel if kernel in self._kernels.keys() else "linear"
        self._z = np.zeros((1, 1)) if device == "cpu" else cp.zeros((1, 1))
        self._d = np.zeros((1, 1) )if device == "cpu" else cp.zeros((1, 1))
        self._loss = 0.0
        self._update = None
        self._hyper_param = hyper_param
        self._device = device

    def _forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = np.dot(self._weights, x) + self._biases
        return self._z

    def _forward_gpu(self, x: cp.ndarray) -> cp.ndarray:
        '''
        Perform forward propagation on the GPU.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = cp.dot(self._weights, x) + self._biases
        return self._z

    def predict(self,
                x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        '''
        Predict the output from the data using the trained support vector machine.
        :param x: Input feature matrix (number samples, number features).
        :return: Support vector machine predictions.
        '''
        samples, features = x.shape

        if self._device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            mx = np.array([[self._kernels[self._kernel]["cpu"](nx[i], nx[j], self._hyper_param)
                            for i in range(samples)] for j in range(samples)])

            return np.where(np.dot(mx, self._weights) + self._biases > 0, 1, -1)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            mx = np.array([[self._kernels[self._kernel]["gpu"](cx[i], cx[j], self._hyper_param)
                            for i in range(samples)] for j in range(samples)])

            p = cp.dot(mx, self._weights) + self._biases

            return cp.where(p > 0, cp.ones_like(p), cp.full_like(p, -1))

    def train(self,
              x: np.ndarray | cp.ndarray,
              y: np.ndarray | cp.ndarray,
              epochs: int) -> list:
        '''
        Train the support vector machine on the data.
        :param x: Input feature matrix (number samples, number features).
        :param y: True value(s).
        :param epochs: Number of training iterations.
        '''
        if x.ndim != 2:
            raise ValueError("SVM @ train: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 2:
            raise ValueError("SVM @ train: parameter 'y' must be a matrix (2-dimensional ndarray).")

        if x.shape[0] != y.shape[0]:
            raise ValueError("SVM @ train: parameter 'x' and 'y' must have the same row count (shape[0]).")

        samples, features = x.shape

        loss = []

        if self._device == "cpu":
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            mx = np.array([[self._kernels[self._kernel]["cpu"](nx[i], nx[j], self._hyper_param)
                            for i in range(samples)] for j in range(samples)])
            self._weights = np.zeros(mx.shape[0])

            for i in range(epochs):
                p = self._forward(mx)

                loss.append(Loss.hinge(ny, p))

                gw = (self._weights - np.multiply(ny, mx.T)).T

                for weight in range(self._weights.shape[0]):
                    gw[weight] = np.where(p >= 1, self._weights[weight], gw[weight])

                gw = np.sum(gw, axis=1)

                gb = -y * self._biases
                gb = np.where(p >= 1, 0, gb)
                gb = np.sum(gb)

                self._weights -= self._learning_rate * gw / samples
                self._biases -= self._learning_rate * gb / samples
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            mx = cp.array([[self._kernels[self._kernel]["gpu"](cx[i], cx[j], self._hyper_param)
                            for i in range(samples)] for j in range(samples)])
            self._weights = cp.zeros(mx.shape[0])

            for i in range(epochs):
                p = self._forward_gpu(mx)

                loss.append(Loss.hinge_gpu(cy, p))

                gw = (self._weights - cp.multiply(cy, mx.T)).T

                for weight in range(self._weights.shape[0]):
                    gw[weight] = cp.where(cp.array(p >= 1), self._weights[weight], gw[weight])

                gw = cp.sum(gw, axis=1)

                gb = -y * self._biases
                gb = cp.where(cp.array(p >= 1), cp.zeros_like(gb), gb)
                gb = cp.sum(gb)

                self._weights -= self._learning_rate * gw / samples
                self._biases -= self._learning_rate * gb / samples

        return loss

    @staticmethod
    def _linear(a: np.ndarray, b: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Calculates the linear kernel trick using dot product of a and b.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        return np.dot(a, b)

    @staticmethod
    def _linear_gpu(a: cp.ndarray, b: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Calculates the linear kernel trick using dot product of a and b on the GPU.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        return cp.dot(a, b)

    @staticmethod
    def _polynomial(a: np.ndarray, b: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Calculates the polynomial kernel trick by raising the dot product of a and b to a power.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters. Uses 'svm_bias' and 'svm_degree'.
        :return: Similarity of a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        r = 0.0
        d = 2.0

        if hyper_param is not None:
            if "svm_bias" in hyper_param.keys():
                r = hyper_param["svm_bias"]

            if "svm_degree" in hyper_param.keys():
                d = hyper_param["svm_degree"]

        return (np.dot(a, b) + r)**d

    @staticmethod
    def _polynomial_gpu(a: cp.ndarray, b: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Calculates the polynomial kernel trick by raising the dot product of a and b to a power on the GPU.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters. Uses 'svm_bias' and 'svm_degree'.
        :return: Similarity of a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        r = 0.0
        d = 2.0

        if hyper_param is not None:
            if "svm_bias" in hyper_param.keys():
                r = hyper_param["svm_bias"]

            if "svm_degree" in hyper_param.keys():
                d = hyper_param["svm_degree"]

        return (cp.dot(a, b) + r)**d

    @staticmethod
    def _rbf(a: np.ndarray, b: np.ndarray, hyper_param: dict = None) -> np.ndarray:
        '''
        Calculates the radial basis function kernel trick by raising e to the negative squared distance between a and b.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters. Uses 'svm_gamma'.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        gamma = 1.0

        if hyper_param is not None and "svm_gamma" in hyper_param.keys():
            gamma = hyper_param["svm_gamma"]

        return np.exp(-gamma * (a - b)**2)

    @staticmethod
    def _rbf_gpu(a: cp.ndarray, b: cp.ndarray, hyper_param: dict = None) -> cp.ndarray:
        '''
        Calculates the radial basis function kernel trick by raising e to the negative squared distance between a and b
        on the GPU.
        :param a: First feature vector (1-dimensional ndarray).
        :param b: Second feature vector (1-dimensional ndarray).
        :param hyper_param: Hyperparameters. Uses 'svm_gamma'.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        gamma = 1.0

        if hyper_param is not None and "svm_gamma" in hyper_param.keys():
            gamma = hyper_param["svm_gamma"]

        return cp.exp(-gamma * (a - b) ** 2)

    @staticmethod
    def _sigmoid(a: np.ndarray, b: np.ndarray, hyper_param: dict=None) -> np.ndarray:
        '''
        Calculates the sigmoid kernel trick by performing the hyperbolic tangent of the dot product of a and b.
        :param a: First feature vector.
        :param b: Second feature vector.
        :param hyper_param: Hyperparameters. Uses 'svm_bias' and 'svm_gamma'.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        r = 0.0
        gamma = 2.0

        if hyper_param is not None:
            if "svm_bias" in hyper_param.keys():
                r = hyper_param["svm_bias"]

            if "svm_gamma" in hyper_param.keys():
                gamma = hyper_param["svm_gamma"]

        return np.tanh(gamma * np.dot(a, b) + r)

    @staticmethod
    def _sigmoid_gpu(a: cp.ndarray, b: cp.ndarray, hyper_param: dict=None) -> cp.ndarray:
        '''
        Calculates the sigmoid kernel trick by performing the hyperbolic tangent of the dot product of a and b on the
        GPU.
        :param a: First feature vector.
        :param b: Second feature vector.
        :param hyper_param: Hyperparameters. Uses 'svm_bias' and 'svm_gamma'.
        :return: Similarity between a and b.
        '''
        if a.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'a' must be a vector (1-dimensional ndarray).")

        if b.ndim != 1:
            raise ValueError("SVM @ _linear: parameter 'b' must be a vector (1-dimensional ndarray).")

        if a.shape[0] != b.shape[0]:
            raise ValueError("SVM @ _linear: parameter 'a' shape[0] must equal parameter 'b' shape[0].")

        r = 0.0
        gamma = 2.0

        if hyper_param is not None:
            if "svm_bias" in hyper_param.keys():
                r = hyper_param["svm_bias"]

            if "svm_gamma" in hyper_param.keys():
                gamma = hyper_param["svm_gamma"]

        return cp.tanh(gamma * cp.dot(a, b) + r)