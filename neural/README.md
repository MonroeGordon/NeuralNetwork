## Neural Module
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

Details:
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

Details:
- **Regularizer**: Class containing the following regularization functions:
  - None: No regularization performed
  - Elastic Net Regerssion
  - LASSO: Least Absolute Shrinkage and Selection Operator (L1)
  - Ridge Regression (L2)

### Optimization Functions (optimizer.py)
Includes a class containing optimization functions.

Details:
- **Optimizer**: Class containing the following optimization functions:
  - None: No optimization performed
  - AdaDelta
  - AdaGrad: Adaptive Gradient
  - Adam: Adaptive Moment Estimation
  - Momentum
  - NAG: Nesterov Accelerated Gradient
  - RMSProp: Root-Mean-Sqaure Propagation
 
### Logistic Regression (logreg.py)
Logistic regression uses the logistic or sigmoid function to calculate the probability of a binary event occurring in one direction or the other. Since logistic regression tries to find optimal coeffiecients for the inputs, it is modeled as a single layer neural network with a single sigmoid output neuron. Using backpropagation, the coefficients are equivalent to weights in the neural network, thereby allowing the model to learn the optimal values using the same regularization and optimization functions used by neural networks. Being a binary classification model, binary cross entropy is used for the loss function, making its negation the log likelihood of the positive binary event occurring.

Details:
- **LogisticRegression**: Creates a logistic regression model as a single layer neural network with a single sigmoid output neuron. The model uses the Adam optimization function and has an optional regularization function that can be set, as well as a tolerance parameter that defines the stopping criteria during training. When the difference between the previous loss value and the current loss value is less than the tolerance value, training stops. After training, the model can predict the probability of output for new inputs. Methods for calculating the log likelihood between true and predicted values and for calculating the accuracy of predictions are available.

### Multilayer Perceptron Neural Network (mlp.py)
A multilayer perceptron neural network is a feedforward neural network with an input layer, hidden layer(s), amd an output layer. Each layer can have a differing number of neurons, but all neurons in a layer, typically, use the same activation function for processing their inputs. All neurons in a layer are connected to the next layer, with each connection having a weight. These weights are adjusted using backpropgation to get the network to align its outputs in the output layer with the true values pertaining to each input it is trained on. After training on a large number of example input/output pairs, a multilayer perceptron neural network can learn the pattern in the data, allowing it to predict the correct outputs for new inputs it was not trained on.

Details:
- **MLPHiddenLayer**: Creates a hidden layer for a multilayer perceptron neural network. A hidden layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, and an optional optimization function for optimizing weight convergence during backpropagation.
- **MLPOutputLayer**: Creates an output layer for a multilayer perceptron neural network. An output layer has a set size, a weight initialization function, an activation function used by all of the layer's neurons, a loss function used to calculate the error between the true outputs and the network's predicted outputs, an optional regularization function for keeping gradients from exploding or vanishing, and an optional optimization function for optimizing weight convergence during backpropagation.
- **MultilayerPerceptronNN**: Creates a full multilayer perceptron neural network with a set input layer size, hidden layer count, hidden layer size(s), and output layer size. The weight initialization, activation, loss, regularization, and optimization functions can be specified for applicable layers. The processing device (cpu/gpu) can also be specified.

### Support Vector Machine (svm.py)
Support vector machines are a supervised learning system that can perform classification or regression by finding hyperplanes that can separate the classes in the data. Support vector machines can use what is known as the kernel trick to transform data into a higher dimension where classes can be separated by hyperplanes. The kernels are non-linear functions such as the polynomial function, radial basis function, and sigmoid function that can handle non-linear data. Since support vector machines train and predict on feature/class data and can use non-linear functions to process the data, they can be modeled as a single layer neural network that does not perform backpropagation, but uses the support vector machine training algorithm. Each input is weighted the same as neural network inputs, along with bias weights, which are updated based on a learning rate and loss function. Predictions are made in the same way as neural networks, by weighting the new inputs to produce an output vector that predicts the classes.

Details:
- **SVM**: Creates a support vector machine as a single layer neural network with a vector of output neurons that apply a specified kernel trick for handling non-linear data, if neeeded. The support vector machine uses the hinge loss function for calculating prediction errors during training.

