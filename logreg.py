from neural.activation import Activation
from neural.initializer import Initializer
from neural.loss import Loss
from neural.optimizer import Optimizer
from neural.regularizer import Regularizer

import cupy as cp
import numpy as np

class LogisticRegression:
    def __init__(self, regularizer_function: str="none", tolerance: float=0.0001, device: str="cpu"):
        '''
        Initializes a logistic regression neural network model.
        :param regularizer_function: Weight regularization function.
        :param tolerance: Tolerance for stopping criteria.
        :param device: CPU or GPU device.
        '''
        self._weights = None
        self._biases = np.zeros((1, 1)) if device == "cpu" else cp.zeros((1, 1))
        self._regularizer_functions = {
            "none": {"cpu": Regularizer.none,
                     "gpu": Regularizer.none_gpu},
            "elastic_net": {"cpu": Regularizer.elastic_net,
                            "gpu": Regularizer.elastic_net_gpu},
            "lasso_l1": {"cpu": Regularizer.lasso_l1,
                         "gpu": Regularizer.lasso_l1_gpu},
            "ridge_l2": {"cpu": Regularizer.ridge_l2,
                         "gpu": Regularizer.ridge_l2_gpu},
        }
        self._regularizer_function = regularizer_function if regularizer_function in self._regularizer_functions \
            else "none"
        self._z = np.zeros((1, 1)) if device == "cpu" else cp.zeros((1, 1))
        self._a = np.zeros((1, 1)) if device == "cpu" else cp.zeros((1, 1))
        self._d = np.zeros((1, 1) )if device == "cpu" else cp.zeros((1, 1))
        self._loss = 0.0
        self._update = None
        self._tolerance = tolerance

    def _forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = np.dot(self._weights, x) + self._biases
        self._a = Activation.sigmoid(self._z)
        return self._a

    def _forward_gpu(self, x: cp.ndarray) -> cp.ndarray:
        '''
        Perform forward propagation on the GPU.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = cp.dot(self._weights, x) + self._biases
        self._a = Activation.sigmoid_gpu(self._z)
        return self._a

    def _backward(self, cycle: int, samples: int, y: np.ndarray, learning_rate: float) -> np.ndarray:
        '''
        Perform backpropagation.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param y: True value(s).
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        loss = (Loss.bin_cross_entropy(y, self._a) +
                self._regularizer_functions[self._regularizer_function]["cpu"](self._weights))
        self._d = Activation.sigmoid_der(self._z, self._a)
        delta = loss * self._d

        self._update, self._weights = Optimizer.adam(
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d)
        self._biases -= (1 / samples) * np.sum(delta) * learning_rate

        return delta

    def _backward_gpu(self, cycle: int, samples: int, y: cp.ndarray, learning_rate: float) -> cp.ndarray:
        '''
        Perform backpropagation on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param y: True value(s).
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        loss = (Loss.bin_cross_entropy_gpu(y, self._a) +
                self._regularizer_functions[self._regularizer_function]["gpu"](self._weights))
        self._d = Activation.sigmoid_der_gpu(self._z, self._a)
        delta = loss * self._d

        self._update, self._weights = Optimizer.adam_gpu(
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d)
        self._biases -= (1 / samples) * cp.sum(delta) * learning_rate

        return delta

    def predict(self,
                x: np.ndarray | cp.ndarray,
                device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Makes predictions using a trained logistic regression neural network model.
        :param x: Input matrix (2-dimensional ndarray).
        :param device: CPU or GPU device/
        :return: Predicted probabilities.
        '''
        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            return self._forward(nx)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            return self._forward_gpu(cx)

    def train(self,
              x: np.ndarray | cp.ndarray,
              y: float | np.ndarray | cp.ndarray,
              device: str="cpu") -> list[float]:
        '''
        Train the logistic regression neural network model.
        :param x: Input matrix (2-dimensional ndarray).
        :param y: True value(s).
        :param device: CPU or GPU device.
        :return: Log likelihood value(s).
        '''
        if x.ndim != 2:
            raise ValueError("LogisticRegression @ train: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 2:
            raise ValueError("LogisticRegression @ train: parameter 'y' must be a matrix (2-dimensional ndarray).")

        if x.shape[0] != y.shape[0]:
            raise ValueError("Logistic Regression @ train: parameter 'x' and 'y' must have the same row count "
                             "(shape[0]).")

        input_size = x.shape[1]
        cycle = 0
        samples = x.shape[0]
        likelihood = []
        prev_loss = 1.0

        if device == "cpu":
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            self._weights = Initializer.glorot_uniform((1, input_size), input_size, 1)
            self._update = np.zeros((1, input_size))

            while prev_loss - self._loss > self._tolerance:
                self._forward(nx)
                self._backward(cycle, samples, ny, 0.001)
                likelihood.append(-self._loss)
                cycle += 1
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            self._weights = Initializer.glorot_uniform_gpu((1, input_size), input_size, 1)
            self._update = cp.zeros((1, input_size))

            while prev_loss - self._loss > self._tolerance:
                self._forward_gpu(cx)
                self._backward_gpu(cycle, samples, cy, 0.001)
                likelihood.append(-self._loss)
                cycle += 1

        return likelihood

    @staticmethod
    def accuracy(y: np.ndarray | cp.ndarray,
                 y_pred: np.ndarray | cp.ndarray,
                 device: str = "cpu") -> float:
        '''
        Calculate the accuracy of the logistic regression predictions.
        :param y: True value(s).
        :param y_pred: Predicted value(s).
        :param device: CPU or GPU device.
        :return: Prediction accuracy value.
        '''
        yp = (y_pred >= 0.5).astype(int)

        if device == "cpu":
            ny = y
            nyp = yp

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            if isinstance(nyp, cp.ndarray):
                nyp = cp.asnumpy(nyp)

            return float(np.mean(np.abs(ny - nyp)))
        else:
            cy = y
            cyp = yp

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            if isinstance(cyp, np.ndarray):
                cyp = cp.asarray(cyp)

            return float(cp.mean(cp.abs(cy - cyp))[0])

    @staticmethod
    def log_likelihood(y: np.ndarray | cp.ndarray,
                       y_pred: np.ndarray | cp.ndarray,
                       device: str = "cpu") -> np.ndarray | cp.ndarray:
        '''
        Calculate the log-likelihood for logistic regression.
        :param y: True value(s).
        :param y_pred: Predicted value(s).
        :param device: CPU or GPU device.
        :return: Log-likelihood value(s).
        '''
        if device == "cpu":
            ny = y
            nyp = y_pred

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            if isinstance(nyp, cp.ndarray):
                nyp = cp.asnumpy(nyp)

            return -Loss.bin_cross_entropy(ny, nyp)
        else:
            cy = y
            cyp = y_pred

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            if isinstance(cyp, np.ndarray):
                cyp = cp.asarray(cyp)

            return -Loss.bin_cross_entropy_gpu(cy, cyp)

    @property
    def coefficients(self) -> np.ndarray | cp.ndarray:
        return self._weights

    @property
    def input_sum(self) -> np.ndarray | cp.ndarray:
        return self._z

    @property
    def intercept(self) -> np.ndarray | cp.ndarray:
        return self._biases

    @property
    def log_loss(self) -> float:
        return self._loss

    @property
    def regularizer_function(self) -> str:
        return self._regularizer_function

    @property
    def y_prediction(self) -> np.ndarray | cp.ndarray:
        return self._a