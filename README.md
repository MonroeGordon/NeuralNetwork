#BlueSkyLib
An AI/machine learning library containing AI mathematical functions and AI models that can be created, trained, and deployed. This library supports CPU and GPU processing.

## Neural
Module containing various neural network architectures that can be created, trained, and deployed.

### Weight Initialization Functions (initializer.py)
Includes a class containing weight initialization functions.

Details:
- **Initializer**: Class containing the following weight initialization functions with uniform or normal distributions:
  - Glorot/Xavier
  - He/Kaiming
  - LeCun

### Activation Functions (activation.py)
Includes a class containing neuron activation functions.

Details:
- **Activation**: Class containing the following neuron activation functions and their derivative functions:
  - Leaky ReLU
  - Linear
  - ReLU
  - Sigmoid
  - Softmax
  - Tanh

### Loss Functions (loss.py)
Includes a class containing loss functions.

- **Loss**: Class containing the following loss functions:
  - Binary Cross-entropy
  - Categorical Cross-entropy
  - Hinge
  - Huber
  - Kullback-Leibler Divergence
  - Mean Absolute Error (L1)
  - Mean Squared Error (L2)
  - Log-cosh
  - Sparse Categorical Cross-entropy
 
### Regularization Functions (regularizer.py)
Includes a class containing regularization functions.

- **Regularizer**: Class containing the following regularization functions:
  - None: No regularization performed
  - Elastic Net Regerssion
  - LASSO: Least Absolute Shrinkage and Selection Operator (L1)
  - Ridge Regression (L2)

### Optimization Functions (optimizer.py)
Includes a class containing optimization functions.

- **Optimizer**: Class containing the following optimization functions:
  - None: No optimization performed
  - AdaDelta
  - AdaGrad: Adaptive Gradient
  - Adam: Adaptive Moment Estimation
  - Momentum
  - NAG: Nesterov Accelerated Gradient
  - RMSProp: Root-Mean-Sqaure Propagation

### Multilayer Perceptron Neural Network (mlp.py)
A multilayer perceptron neural network is a feed forward neural network with an input layer, hidden layer(s), amd an output layer. Each layer can have a differing number of neurons, but all neurons in a layer, typically, use the same activation function for processing their inputs. All neurons in a layer are connected to the next layer, with each connection having a weight. These weights are adjusted using backpropgation to get the network to align its outputs in the output layer with the true values pertaining to each input it is trained on. After training on a large number of example input/output pairs, a multilayer perceptron neural network can learn the pattern in the data, allowing it to predict the correct outputs for new inputs it was not trained on.

Details:
- **MLPHiddenLayer**: Creates a hidden layer for a multilayer perceptron neural network. A hidden layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, and an optional optimization function for optimizing weight convergence during backpropagation.
- **MLPOutputLayer**: Creates an output layer for a multilayer perceptron neural network. An output layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, a loss function used to calculate the error between the true outputs and the network's predicted outputs, an optional regularization function for keeping gradients from exploding or vanishing, and an optional optimization function for optimizing weight convergence during backpropagation.
- **MultilayerPerceptronNN**: Creates a full multilayer perceptron neural network with a set input layer size, hidden layer count, hidden layer size(s), and output layer size. The weight initialization, activation, loss, regularization, and optimization functions can be specified for applicable layers. The processing device (cpu/gpu) can also be specified.
