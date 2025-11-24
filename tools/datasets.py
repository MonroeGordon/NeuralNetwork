import cupy as cp
import numpy as np
import pandas
import pandas as pd

class Data:
    '''
    Data class. Stores features and target data from a dataset and provides methods for accessing and using the data.
    '''

    def __init__(self,
                 x_names: np.ndarray,
                 y_names: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray):
        '''
        Initializes a Data class with the specified feature and target data.
        :param x_names: Feature names (number of features).
        :param y_names: Class names (number of classes).
        :param x: Input feature matrix (number samples, number features).
        :param y: Target value(s) (target values, class indices, or one hot encoded).
        '''
        if x.ndim != 2:
            raise ValueError("Data: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("Data: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        if x_names.ndim != 1:
            raise ValueError("Data: parameter 'x_names' must be a vector (1-dimensional ndarray).")

        if x_names.shape[0] != x.shape[1]:
            raise ValueError("Data: parameter 'x_names' shape[0] must equal parameter 'x' shape[1].")

        if y_names.ndim != 1:
            raise ValueError("Data: parameter 'y_names' must be a vector (1-dimensional ndarray).")

        if y.ndim == 1:
            if len(np.unique(y)) != y_names.shape[0]:
                raise ValueError("Data: parameter 'y' unique value count must equal parameter 'y_names' shape[0].")
        else:
            if y.shape[1] != y_names.shape[0]:
                raise ValueError("Data: parameter 'y' shape[1] must equal parameter 'y_names' shape[0].")

        self._feature_names = x_names
        self._class_names = y_names
        self._features = x
        self._targets = y

    @property
    def class_names(self):
        return self._class_names

    @ property
    def feature_names(self):
        return self._feature_names

    @property
    def features(self):
        return self._features

    @property
    def features_gpu(self):
        return cp.asarray(self._features)

    @property
    def targets(self):
        return self._targets

    @property
    def targets_gpu(self):
        return cp.asarray(self._targets)

class Datasets:
    '''
    Datasets class. Provides methods for accessing dataset files and converting them into data that a machine learning/
    AI model can process.
    '''

    DATASET_PATH = "datasets/"

    @staticmethod
    def load_iris(one_hot: bool=False) -> Data:
        '''
        Loads the Iris dataset into a Data class.
        :param one_hot: Perform one hot encoding of target classes.
        :return: A Data class containing the Iris dataset.
        '''
        df = pandas.read_csv(Datasets.DATASET_PATH + "iris.csv")
        fdf = df.iloc[:, :4]
        tdf = df.iloc[:, 4]

        x = fdf.to_numpy()
        y = tdf.to_numpy()

        x_names = np.array(["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"])

        y_names = np.unique(y)

        if one_hot:
            tdf = pd.get_dummies(tdf, columns=[4])
            y = tdf.to_numpy()
        else:
            for i in range(len(y_names)):
                indices = np.where(y == y_names[i])
                y[indices] = i

        return Data(x_names, y_names, x, y)

    @staticmethod
    def create_regression(n_samples: int=100,
                          n_features: int=1,
                          n_informative: int=1,
                          n_targets: int=1,
                          bias: list=None,
                          noise: float=1.0,
                          random_state: int=None,
                          distribution: str="uniform",
                          coefficients: bool=False) -> Data | tuple:
        '''
        Creates a synthetic regression dataset.
        :param n_samples: Number of data samples to generate.
        :param n_features: Number of features to generate.
        :param n_informative: Number of features contributing to the target variable(s).
        :param n_targets: Number of target variables.
        :param bias: The intercept term(s) in the linear model.
        :param noise: Standard deviation of the noise generated for the data.
        :param random_state: Random seed for reproducibility.
        :param distribution: Uniform or normal distribution for random feature values.
        :param coefficients: Return the true coefficients of the linear model.
        :return: A Data class containing the synthetic regression dataset.
        '''
        if n_samples < 1:
            raise ValueError("Datasets @ create_regression: parameter 'n_samples' needs to exceed 0.")

        if n_features < 1:
            raise ValueError("Datasets @ create_regression: parameter 'n_features' needs to exceed 0.")

        if n_informative < 1:
            raise ValueError("Datasets @ create_regression: parameter 'n_informative' needs to exceed 0.")

        if n_targets < 1:
            raise ValueError("Datasets @ create_regression: parameter 'n_targets' needs to exceed 0.")

        if bias is not None and len(bias) != n_targets:
            raise ValueError("Datasets @ create_regression: parameter 'bias' must be None or have a length equal to "
                             "parameter 'n_targets'.")

        if not noise > 0.0:
            raise ValueError("Datasets @ create_regression: parameter 'noise' needs to exceed 0.0.")

        if distribution != "uniform" and distribution != "normal":
            raise ValueError("Datasets @ create_regression: parameter 'distribution' needs to be 'uniform' or "
                             "'normal'.")

        if n_informative > n_features:
            raise ValueError("Datasets @ create_regression: parameter 'n_informative' cannot exceed parameter "
                             "'n_features'.")

        np.random.seed(random_state)

        if distribution == "uniform":
            x = np.random.uniform(low=-10.0, high=10.0, size=(n_samples, n_features))
        else:
            x = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

        c = np.random.normal(loc=0.0, scale=1.0, size=(n_informative, n_targets))
        xi = [np.random.choice(n_features, size=n_informative, replace=False) for _ in range(n_targets)]

        n = np.random.normal(loc=0.0, scale=noise, size=(n_targets, n_samples))

        b = np.zeros(n_targets) if bias is None else np.array(bias)

        y = np.array([(x[:, xi[i]] @ c[:, i] + b[i] + n[i, :])
                      for i in range(n_targets)]).reshape((n_samples, n_targets))

        x_names = []
        y_names = []

        for i in range(n_features):
            x_names.append(f"Feature {i}")

        for i in range(n_targets):
            y_names.append(f"Class {i}")

        x_names = np.array(x_names)
        y_names = np.array(y_names)

        if coefficients:
            return Data(x_names, y_names, x, y), c
        else:
            return Data(x_names, y_names, x, y)