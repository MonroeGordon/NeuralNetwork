import cupy as cp
import numpy as np

def accuracy(y: np.ndarray | cp.ndarray,
             y_pred: np.ndarray | cp.ndarray,
             device: str = "cpu") -> float:
    '''
    Calculate the accuracy of the predictions.
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