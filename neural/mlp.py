from activation import Activation
from initializer import Initializer
from loss import Loss
from optimizer import Optimizer
from regularizer import Regularizer

import cupy as cp
import numpy as np

class MLPHiddenLayer:
    def __init__(self,
                 size: int,
                 initializer_function: str="he_uniform",
                 activation_function: str="relu",
                 optimizer_function: str="none"):
        '''
        Initialize a multilayer perceptron hidden layer with the given size and activation function.
        :param size: Layer size.
        :param initializer_function: Weight initializer function.
        :param activation_function: Activation function.
        :param optimizer_function: Optimizer function.
        '''
        if size < 1:
            raise ValueError("MLPHiddenLayer: size must exceed 0.")

        self._size = size
        self._weights = None
        self._biases = None
        self._initializer_functions = {
            "glorot_normal": {"cpu": Initializer.glorot_normal,
                              "gpu": Initializer.glorot_normal_gpu},
            "glorot_uniform": {"cpu": Initializer.glorot_uniform,
                               "gpu": Initializer.glorot_uniform_gpu},
            "he_normal": {"cpu": Initializer.he_normal,
                          "gpu": Initializer.he_normal_gpu},
            "he_uniform": {"cpu": Initializer.he_uniform,
                           "gpu": Initializer.he_uniform_gpu},
            "lecun_normal": {"cpu": Initializer.lecun_normal,
                             "gpu": Initializer.lecun_normal_gpu},
            "lecun_uniform": {"cpu": Initializer.lecun_uniform,
                              "gpu": Initializer.lecun_uniform_gpu}
        }
        self._activation_functions = {
            "leaky_relu": {"cpu": (Activation.leaky_relu,
                                   Activation.leaky_relu_der),
                           "gpu": (Activation.leaky_relu_gpu,
                                   Activation.leaky_relu_der_gpu)},
            "relu": {"cpu": (Activation.relu,
                             Activation.relu_der),
                     "gpu": (Activation.relu_gpu,
                             Activation.relu_der_gpu)},
            "sigmoid": {"cpu": (Activation.sigmoid,
                                Activation.sigmoid_der),
                        "gpu": (Activation.sigmoid_gpu,
                                Activation.sigmoid_der_gpu)},
            "tanh": {"cpu": (Activation.tanh,
                             Activation.tanh_der),
                     "gpu": (Activation.tanh_gpu,
                             Activation.tanh_der_gpu)},
        }
        self._optimizer_functions = {
            "none": {"cpu": Optimizer.none,
                     "gpu": Optimizer.none_gpu},
            "adadelta": {"cpu": Optimizer.adadelta,
                         "gpu": Optimizer.adadelta_gpu},
            "adagrad": {"cpu": Optimizer.adagrad,
                        "gpu": Optimizer.adagrad_gpu},
            "adam": {"cpu": Optimizer.adam,
                     "gpu": Optimizer.adam_gpu},
            "momentum": {"cpu": Optimizer.momentum,
                         "gpu": Optimizer.momentum_gpu},
            "nesterov": {"cpu": Optimizer.nesterov,
                         "gpu": Optimizer.nesterov_gpu},
            "rmsprop": {"cpu": Optimizer.rmsprop,
                        "gpu": Optimizer.rmsprop_gpu},
        }
        self._initializer_function = initializer_function if initializer_function in self._initializer_functions \
            else "he_uniform"
        self._activation_function = activation_function if activation_function in self._activation_functions else "relu"
        self._optimizer_function = optimizer_function if optimizer_function in self._optimizer_functions else "none"
        self._hyper_param = {
            "leaky_relu_alpha": 0.01
        }
        self._z = None
        self._a = None
        self._d = None
        self._update = None

    def init_layer(self, prev_layer_size: int):
        '''
        Initialize the layer with the given previous layer's size and this layer's size.
        :param prev_layer_size: Previous layer's size.
        '''
        self._weights = self._initializer_functions[self._initializer_function]["cpu"](
            (self._size, prev_layer_size), prev_layer_size, self._size)
        self._biases = np.zeros((self._size, 1))
        self._z = np.zeros((self._size, 1))
        self._a = np.zeros((self._size, 1))
        self._d = np.zeros((self._size, 1))
        self._update = np.zeros((self._size, prev_layer_size))

    def init_layer_gpu(self, prev_layer_size: int):
        '''
        Initialize the layer with the given previous layer's size and this layer's size on the GPU.
        :param prev_layer_size: Previous layer's size.
        '''
        self._weights = self._initializer_functions[self._initializer_function]["gpu"](
            (self._size, prev_layer_size), prev_layer_size, self._size)
        self._biases = cp.zeros((self._size, 1))
        self._z = cp.zeros((self._size, 1))
        self._a = cp.zeros((self._size, 1))
        self._d = cp.zeros((self._size, 1))
        self._update = cp.zeros((self._size, prev_layer_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = np.dot(self._weights, x) + self._biases
        self._a = self._activation_functions[self._activation_function]["cpu"][0](self._z, self._hyper_param)
        return self._a

    def forward_gpu(self, x: cp.ndarray) -> cp.ndarray:
        '''
        Perform forward propagation on the GPU.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = cp.dot(self._weights, x) + self._biases
        self._a = self._activation_functions[self._activation_function]["gpu"][0](self._z, self._hyper_param)
        return self._a

    def backward(self,
                 cycle: int,
                 samples: int,
                 w: np.ndarray,
                 delta: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        '''
        Perform backpropagation.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param w: Previous layer weights.
        :param delta: Backpropagated error value.
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        self._d = self._activation_functions[self._activation_function]["cpu"][1](self._z, self._a, self._hyper_param)
        delta = np.dot(w.T, delta) * self._d

        self._update, self._weights = self._optimizer_functions[self._optimizer_function]["cpu"](
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d, self._hyper_param)
        self._biases -= (1 / samples) * np.sum(delta) * learning_rate

        return delta

    def backward_gpu(self,
                     cycle: int,
                     samples: int,
                     w: cp.ndarray,
                     delta: cp.ndarray,
                     learning_rate: float) -> cp.ndarray:
        '''
        Perform backpropagation on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param w: Previous layer weights.
        :param delta: Backpropagated error value.
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        self._d = self._activation_functions[self._activation_function]["gpu"][1](self._z, self._a, self._hyper_param)
        delta = cp.dot(w.T, delta) * self._d

        self._update, self._weights = self._optimizer_functions[self._optimizer_function]["gpu"](
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d, self._hyper_param)
        self._biases -= (1 / samples) * cp.sum(delta) * learning_rate

        return delta

    @property
    def activation_function(self) -> str:
        return self._activation_function

    @property
    def activations(self) -> np.ndarray | cp.ndarray:
        return self._a

    @property
    def biases(self) -> np.ndarray | cp.ndarray:
        return self._biases

    @property
    def size(self) -> int:
        return self._size

    @property
    def weighted_sums(self) -> np.ndarray | cp.ndarray:
        return self._z

    @property
    def weights(self) -> np.ndarray | cp.ndarray:
        return self._weights

class MLPOutputLayer:
    def __init__(self,
                 size: int,
                 initializer_function: str="glorot_uniform",
                 activation_function: str="softmax",
                 loss_function: str="cat_cross_entropy",
                 regularizer_function: str="none",
                 optimizer_function: str="none"):
        '''
        Initialize a multilayer perceptron output layer with the given size, activation function and loss function.
        :param size: Layer size.
        :param activation_function: Activation function.
        :param loss_function: Loss function.
        :param regularizer_function: Regularizer function.
        :param optimizer_function: Optimizer function.
        '''
        if size < 1:
            raise ValueError("MLPOutputLayer: size must exceed 0.")

        self._size = size
        self._weights = None
        self._biases = None
        self._initializer_functions = {
            "glorot_normal": {"cpu": Initializer.glorot_normal,
                              "gpu": Initializer.glorot_normal_gpu},
            "glorot_uniform": {"cpu": Initializer.glorot_uniform,
                               "gpu": Initializer.glorot_uniform_gpu},
            "he_normal": {"cpu": Initializer.he_normal,
                          "gpu": Initializer.he_normal_gpu},
            "he_uniform": {"cpu": Initializer.he_uniform,
                           "gpu": Initializer.he_uniform_gpu},
            "lecun_normal": {"cpu": Initializer.lecun_normal,
                             "gpu": Initializer.lecun_normal_gpu},
            "lecun_uniform": {"cpu": Initializer.lecun_uniform,
                              "gpu": Initializer.lecun_uniform_gpu}
        }
        self._activation_functions = {
            "linear": {"cpu": (Activation.linear,
                               Activation.linear_der),
                       "gpu": (Activation.linear_gpu,
                               Activation.linear_der_gpu)},
            "sigmoid": {"cpu": (Activation.sigmoid,
                                Activation.sigmoid_der),
                        "gpu": (Activation.sigmoid_gpu,
                                Activation.sigmoid_der_gpu)},
            "softmax": {"cpu": (Activation.softmax,
                                Activation.softmax_der),
                        "gpu": (Activation.softmax_gpu,
                                Activation.softmax_der_gpu)},
        }
        self._loss_functions = {
            "bin_cross_entropy": {"cpu": Loss.bin_cross_entropy,
                                  "gpu": Loss.bin_cross_entropy_gpu},
            "cat_cross_entropy": {"cpu": Loss.cat_cross_entropy,
                                  "gpu": Loss.cat_cross_entropy_gpu},
            "hinge": {"cpu": Loss.hinge,
                      "gpu": Loss.hinge_gpu},
            "huber": {"cpu": Loss.huber,
                      "gpu": Loss.huber_gpu},
            "kl_divergence": {"cpu": Loss.kl_divergence,
                              "gpu": Loss.kl_divergence_gpu},
            "l1_mae": {"cpu": Loss.l1_mae,
                       "gpu": Loss.l1_mae_gpu},
            "l2_mse": {"cpu": Loss.l2_mse,
                       "gpu": Loss.l2_mse_gpu},
            "log_cosh": {"cpu": Loss.log_cosh,
                         "gpu": Loss.log_cosh_gpu},
            "sparse_cat_cross_entropy": {"cpu": Loss.sparse_cat_cross_entropy,
                                         "gpu": Loss.sparse_cat_cross_entropy_gpu},
        }
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
        self._optimizer_functions = {
            "none": {"cpu": Optimizer.none,
                     "gpu": Optimizer.none_gpu},
            "adadelta": {"cpu": Optimizer.adadelta,
                         "gpu": Optimizer.adadelta_gpu},
            "adagrad": {"cpu": Optimizer.adagrad,
                        "gpu": Optimizer.adagrad_gpu},
            "adam": {"cpu": Optimizer.adam,
                     "gpu": Optimizer.adam_gpu},
            "momentum": {"cpu": Optimizer.momentum,
                         "gpu": Optimizer.momentum_gpu},
            "nesterov": {"cpu": Optimizer.nesterov,
                         "gpu": Optimizer.nesterov_gpu},
            "rmsprop": {"cpu": Optimizer.rmsprop,
                        "gpu": Optimizer.rmsprop_gpu},
        }
        self._initializer_function = initializer_function if initializer_function in self._initializer_functions \
            else "glorot_uniform"
        self._activation_function = activation_function if activation_function in self._activation_functions \
            else "softmax"
        self._loss_function = loss_function if loss_function in self._loss_functions else "cat_cross_entropy"
        self._regularizer_function = regularizer_function if regularizer_function in self._regularizer_functions \
            else "none"
        self._optimizer_function = optimizer_function if optimizer_function in self._optimizer_functions else "none"
        self._hyper_param = {}
        self._z = None
        self._a = None
        self._d = None
        self._loss = 0.0
        self._update = None
        
    def init_layer(self, prev_layer_size: int):
        '''
        Initialize the layer with the given previous layer's size and this layer's size.
        :param prev_layer_size: Previous layer's size.
        '''
        self._weights = self._initializer_functions[self._initializer_function]["cpu"](
            (self._size, prev_layer_size), prev_layer_size, self._size)
        self._biases = np.zeros((self._size, 1))
        self._z = np.zeros((self._size, 1))
        self._a = np.zeros((self._size, 1))
        self._update = np.zeros((self._size, prev_layer_size))

    def init_layer_gpu(self, prev_layer_size: int):
        '''
        Initialize the layer with the given previous layer's size and this layer's size on the GPU.
        :param prev_layer_size: Previous layer's size.
        '''
        self._weights = self._initializer_functions[self._initializer_function]["gpu"](
            (self._size, prev_layer_size), prev_layer_size, self._size)
        self._biases = cp.zeros((self._size, 1))
        self._z = cp.zeros((self._size, 1))
        self._a = cp.zeros((self._size, 1))
        self._update = cp.zeros((self._size, prev_layer_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = np.dot(self._weights, x) + self._biases
        self._a = self._activation_functions[self._activation_function]["cpu"][0](self._z, self._hyper_param)
        return self._a

    def forward_gpu(self, x: cp.ndarray) -> cp.ndarray:
        '''
        Perform forward propagation on the GPU.
        :param x: Input data.
        :return: Output of the layer.
        '''
        self._z = cp.dot(self._weights, x) + self._biases
        self._a = self._activation_functions[self._activation_function]["gpu"][0](self._z, self._hyper_param)
        return self._a

    def backward(self, cycle: int, samples: int, y: np.ndarray, learning_rate: float) -> np.ndarray:
        '''
        Perform backpropagation.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param y: True value(s).
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        loss = (self._loss_functions[self._loss_function]["cpu"](y, self._a, self._hyper_param) +
                self._regularizer_functions[self._regularizer_function]["cpu"](self._weights, self._hyper_param))
        self._d = self._activation_functions[self._activation_function]["cpu"][1](self._z, self._a, self._hyper_param)
        delta = loss * self._d

        self._update, self._weights = self._optimizer_functions[self._optimizer_function]["cpu"](
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d, self._hyper_param)
        self._biases -= (1 / samples) * np.sum(delta) * learning_rate

        return delta

    def backward_gpu(self, cycle: int, samples: int, y: cp.ndarray, learning_rate: float) -> cp.ndarray:
        '''
        Perform backpropagation on the GPU.
        :param cycle: Training cycle.
        :param samples: Number of training samples.
        :param y: True value(s).
        :param learning_rate: Learning rate.
        :return: This layer's error value.
        '''
        loss = (self._loss_functions[self._loss_function]["gpu"](y, self._a, self._hyper_param) +
                self._regularizer_functions[self._regularizer_function]["gpu"](self._weights, self._hyper_param))
        self._d = self._activation_functions[self._activation_function]["gpu"][1](self._z, self._a, self._hyper_param)
        delta = loss * self._d

        self._update, self._weights = self._optimizer_functions[self._optimizer_function]["gpu"](
            cycle, samples, self._update, learning_rate, delta, self._weights, self._a, self._d, self._hyper_param)
        self._biases -= (1 / samples) * cp.sum(delta) * learning_rate

        return delta

    @property
    def activation_function(self) -> str:
        return self._activation_function

    @property
    def activations(self) -> np.ndarray | cp.ndarray:
        return self._a

    @property
    def biases(self) -> np.ndarray | cp.ndarray:
        return self._biases

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def loss_function(self) -> str:
        return self._loss_function

    @property
    def regularizer_function(self) -> str:
        return self._regularizer_function

    @property
    def size(self) -> int:
        return self._size

    @property
    def weighted_sums(self) -> np.ndarray | cp.ndarray:
        return self._z

    @property
    def weights(self) -> np.ndarray | cp.ndarray:
        return self._weights

class MultilayerPerceptronNN:
    def __init__(self,
                 input_size: int,
                 hidden_size: list[int],
                 output_size: int,
                 hidden_activation: list[str]=None,
                 output_activation: str="softmax",
                 loss: str="cat_cross_entropy",
                 regularizer: str="none",
                 learning_rate=0.01,
                 device="cpu"):
        '''
        Initialize the neural network with the given parameters.
        :param input_size: Input layer size.
        :param hidden_size: List of hidden layer sizes.
        :param output_size: Output layer size.
        :param hidden_activation: List of hidden layer activation functions.
        :param output_activation: Output layer activation function.
        :param loss: Loss function.
        :param regularizer: Regularizer function.
        :param learning_rate: Learning rate for weight updates.
        :param device: CPU or GPU processing device.
        '''
        if input_size < 1:
            raise ValueError("MultilayerPerceptronNN: input size must exceed 0.")
        if output_size < 1:
            raise ValueError("MultilayerPerceptronNN: output size must exceed 0.")

        if learning_rate <= 0.0:
            raise ValueError("MultilayerPerceptronNN: learning rate must exceed 0.0.")

        if hidden_activation is None:
            hidden_activation = ["relu" for size in hidden_size]
        elif len(hidden_activation) != len(hidden_size):
            raise ValueError("MultilayerPerceptronNN: length of hidden layer sizes does not match length of hidden "
                             "layer activations.")

        if device != "cpu" and device != "gpu":
            raise ValueError("MultilayerPerceptronNN: device must be cpu or gpu.")

        self._device = device
        self._cycle = 0
        self._learning_rate = learning_rate
        self._input_size = input_size
        self._hidden_layer = [MLPHiddenLayer(s, a) for s, a in zip(hidden_size, hidden_activation)]
        self._output_layer = MLPOutputLayer(output_size, output_activation, loss, regularizer)
        self._activations = []
        self._deltas = []

        if device == "cpu":
            self._hidden_layer[0].init_layer(self._input_size)

            for i in range(1, len(self._hidden_layer)):
                self._hidden_layer[i].init_layer(hidden_size[i - 1])

            self._output_layer.init_layer(hidden_size[-1])
        else:
            self._hidden_layer[0].init_layer_gpu(self._input_size)

            for i in range(1, len(self._hidden_layer)):
                self._hidden_layer[i].init_layer_gpu(hidden_size[i - 1])

            self._output_layer.init_layer_gpu(hidden_size[-1])

    def _forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the neural network.
        '''
        self._activations = [x]

        for h in self._hidden_layer:
            self._activations.append(h.forward(self._activations[-1]))

        self._activations.append(self._output_layer.forward(self._activations[-1]))

        return self._activations[-1]

    def _forward_gpu(self, x: cp.ndarray) -> cp.ndarray:
        '''
        Perform forward propagation.
        :param x: Input data.
        :return: Output of the neural network.
        '''
        self._activations = [x]

        for h in self._hidden_layer:
            self._activations.append(h.forward_gpu(self._activations[-1]))

        self._activations.append(self._output_layer.forward_gpu(self._activations[-1]))

        return self._activations[-1]

    def _backward(self, y: np.ndarray):
        '''
        Perform backward propagation.
        :param y: True output labels.
        '''
        samples = y.shape[0]
        self._deltas = []

        # Compute error at output layer
        self._deltas.append(self._output_layer.backward(self._cycle, samples, y, self._learning_rate))

        # Backpropagate
        for l in range(1, len(self._hidden_layer)):
            w = self._hidden_layer[-l + 1].weights if l > 1 else self._output_layer.weights
            self._deltas.append(self._hidden_layer[-l].backward(
                self._cycle, samples, w, self._deltas[-1], self._learning_rate))

    def _backward_gpu(self, y: cp.ndarray):
        '''
        Perform backward propagation on the GPU.
        :param y: True output labels.
        '''
        samples = y.shape[0]
        self._deltas = []

        # Compute error at output layer
        self._deltas.append(self._output_layer.backward_gpu(self._cycle, samples, y, self._learning_rate))

        # Backpropagate
        for l in range(1, len(self._hidden_layer)):
            w = self._hidden_layer[-l + 1].weights if l > 1 else self._output_layer.weights
            self._deltas.append(self._hidden_layer[-l].backward(
                self._cycle, samples, w, self._deltas[-1], self._learning_rate))

    def train(self,
              x: np.ndarray | cp.ndarray,
              y: np.ndarray | cp.ndarray,
              epochs: int) -> list[float]:
        '''
        Train the neural network.
        :param x: Input data.
        :param y: True output labels.
        :param epochs: Number of training iterations.
        :return: list of loss values for each epoch.
        '''
        loss = []

        if self._device == "cpu":
            if not isinstance(x, np.ndarray):
                raise TypeError("MultilayerPerceptronNN @ train: x must be a numpy ndarray when using cpu device.")

            if not isinstance(y, np.ndarray):
                raise TypeError("MultilayerPerceptronNN @ train: y must be a numpy ndarray when using cpu device.")

            for epoch in range(epochs):
                self._forward(x)
                self._backward(y)
                loss.append(self._output_layer.loss)
                self._cycle += 1
        else:
            if not isinstance(x, cp.ndarray):
                raise TypeError("MultilayerPerceptronNN @ train: x must be a cupy ndarray when using gpu device.")

            if not isinstance(y, cp.ndarray):
                raise TypeError("MultilayerPerceptronNN @ train: y must be a cupy ndarray when using gpu device.")

            for epoch in range(epochs):
                self._forward_gpu(x)
                self._backward_gpu(y)
                loss.append(self._output_layer.loss)
                self._cycle += 1

        return loss

    def test(self, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        '''
        Test the neural network.
        :param x: Input data.
        '''
        if self._device == "cpu":
            if not isinstance(x, np.ndarray):
                raise TypeError("MultilayerPerceptronNN @ test: x must be a numpy ndarray when using cpu device.")

            return self._forward(x)
        else:
            if not isinstance(x, cp.ndarray):
                raise TypeError("MultilayerPerceptronNN @ test: x must be a cupy ndarray when using gpu device.")

            return self._forward_gpu(x)