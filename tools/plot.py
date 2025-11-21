import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

def plot_ols(x: np.ndarray | cp.ndarray,
             y: np.ndarray | cp.ndarray,
             y_pred: np.ndarray | cp.ndarray):
    '''
    Plot ordinary least squares (OLS) regression prediction.
    :param x: Design matrix without the intercept term.
    :param y: Response variable vector.
    :param y_pred: Predicted values.
    '''
    nx = x
    ny = y
    nyp = y_pred

    if isinstance(nx, cp.ndarray):
        nx = cp.asnumpy(x)

    if isinstance(ny, cp.ndarray):
        ny = cp.asnumpy(y)

    if isinstance(nyp, cp.ndarray):
        nyp = cp.asnumpy(nyp)

    plt.scatter(nx, ny, color='blue', label='Data Points')
    plt.plot(nx, nyp, color='red', label='OLS Fit')
    plt.xlabel('Predictor Variable (x)')
    plt.ylabel('Response Variable (y)')
    plt.title('Ordinary Least Squares (OLS) Regression')
    plt.legend()
    plt.show()