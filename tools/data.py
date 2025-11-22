import cupy as cp
import numpy as np

def class_proportions(y: np.ndarray | cp.ndarray) -> np.ndarray:
    '''
    Return an array containing the proportion of the dataset each class takes up.
    :param y: Target values.
    :return: Proportions of class values.
    '''
    ny = y

    if isinstance(y, cp.ndarray):
        ny = cp.asnumpy(ny)

    classes = class_values(ny)

    total_count = len(ny)
    class_count = []

    for i in range(len(classes)):
        class_count.append(len(np.where(ny == classes[i])[0]))
        class_count[-1] /= total_count

    return np.array(class_count)

def class_values(y: np.ndarray | cp.ndarray) -> np.ndarray:
    '''
    Return an array containing each unique class value.
    :param y: Target values.
    :return: Unique class values.
    '''
    ny = y

    if isinstance(y, cp.ndarray):
        ny = cp.asnumpy(ny)

    return np.unique(ny)

def pca_data_reconstruct(x_reduced: np.ndarray | cp.ndarray,
                         eigenvectors: np.ndarray | cp.ndarray,
                         device: str="cpu") -> np.ndarray | cp.ndarray:
    '''
    Reconstruct the data from dimensions reduced by principal component analysis (PCA).
    :param x_reduced: Data with dimension reduced by PCA.
    :param eigenvectors: Eigenvectors corresponding to the components used.
    :param device: CPU or GPU device.
    :return: Reconstructed data.
    '''
    if device == "cpu":
        nx = x_reduced
        ne = eigenvectors

        if isinstance(nx, cp.ndarray):
            nx = cp.asnumpy(nx)

        if isinstance(ne, cp.ndarray):
            ne = cp.asnumpy(ne)

        return np.dot(nx, ne.T) + np.mean(nx, axis=0)
    else:
        cx = x_reduced
        ce = eigenvectors

        if isinstance(cx, np.ndarray):
            cx = cp.asarray(cx)

        if isinstance(ce, np.ndarray):
            ce = cp.asarray(ce)

        return cp.dot(cx, ce.T) + cp.mean(cx, axis=0)

def train_test_split(x: np.ndarray | cp.ndarray,
                     y: np.ndarray | cp.ndarray,
                     test_size: float | int=0.25,
                     random_state: int=0,
                     stratify: bool=False,
                     shuffle: bool=True) -> tuple:
    '''
    Split x and y data into training and test sets.
    :param x: Input features.
    :param y: Target values.
    :param test_size: Proportion of or number of samples to include in test sets.
    :param random_state: Random seed for reproducibility.
    :param stratify: Split data so class proportions in training and test sets to match original dataset proportions.
    :param shuffle: Shuffle the data before splitting.
    :return: Training and test data sets for x and y.
    '''
    if x.shape[0] != y.shape[0]:
        raise ValueError("@ train_test_split: parameters 'x' and 'y' must have same number of rows (shape[0]).")

    if test_size <= 0:
        raise ValueError("@ train_test_split: parameter 'test_size' must exceed 0.")

    if test_size >= len(y) - 1:
        raise ValueError("@ train_test_split: parameter 'test_size' must be less than the length of parameter 'y'.")

    nx = x
    ny = y

    if isinstance(nx, cp.ndarray):
        nx = cp.asnumpy(nx)

    if isinstance(ny, cp.ndarray):
        ny = cp.asnumpy(ny)

    if shuffle:
        rng = np.random.default_rng(random_state)
        shuffled_indices = rng.permutation(len(ny))
        nx = nx[shuffled_indices]
        ny = ny[shuffled_indices]

    if stratify:
        nxs = []
        nys = []

        classes = class_values(ny)
        class_count = len(classes)
        class_prop = class_proportions(ny)
        test_count = []

        x_train = np.zeros((0, nx.shape[1]))
        x_test = np.zeros((0, nx.shape[1]))
        y_train = np.zeros(0)
        y_test = np.zeros(0)

        for i in range(class_count):
            indices = np.where(ny == classes[i])[0]

            nxs.append(nx[indices])
            nys.append(ny[indices])

            if test_size > 1:
                test_count.append(test_size * class_prop[i])
            else:
                test_count.append(len(ny) * test_size * class_prop[i])

            x_train = np.concatenate((x_train, nxs[-1][test_count[-1]:]), axis=0)
            x_test = np.concatenate((x_test, nxs[-1][:test_count[-1]]), axis=0)
            y_train = np.concatenate((y_train, nys[-1][test_count[-1]:]), axis=0)
            y_test = np.concatenate((y_test, nys[-1][:test_count[-1]]), axis=0)

        return x_train, x_test, y_train, y_test
    else:
        test_count = test_size if test_size > 1 else test_size * len(ny)

        x_train = nx[test_count:]
        x_test = nx[:test_count]
        y_train = ny[test_count:]
        y_test = ny[:test_count]

        return x_train, x_test, y_train, y_test