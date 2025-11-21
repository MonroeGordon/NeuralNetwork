from collections import Counter
from linalg.functions import euclidean_dist
from tools.metrics import accuracy
from tools.data import train_test_split

import cupy as cp
import numpy as np

class KNearestNeighbor:
    @staticmethod
    def predict(x_train: np.ndarray | cp.ndarray,
                y_train: list,
                x_test: np.ndarray | cp.ndarray,
                k: int,
                device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Predict the class label for the test data using K-nearest neighbors.
        :param x_train: Training data features.
        :param y_train: Training data labels.
        :param x_test: Test data features.
        :param k: Number of neighbors to consider.
        :param device: CPU or GPU device.
        :return: Predicted labels for the test data.
        '''
        predictions = []

        if device == "cpu":
            nx_train = x_train
            nx_test = x_test

            if isinstance(nx_train, cp.ndarray):
                nx_train = cp.asnumpy(nx_train)

            if isinstance(nx_test, cp.ndarray):
                nx_test = cp.asnumpy(nx_test)

            for i in range(nx_test.shape[0]):
                test_points = np.full(nx_train.shape[0], nx_test[i])
                distances = euclidean_dist(test_points, nx_train)

                k_indices = np.argsort(distances)[:k]

                k_nearest_labels = [y_train[i] for i in k_indices]

                most_common = Counter(k_nearest_labels).most_common(1)
                predictions.append(most_common[0][0])

            return np.array(predictions)
        else:
            cx_train = x_train
            cx_test = x_test

            if isinstance(cx_train, np.ndarray):
                cx_train = cp.asarray(cx_train)

            if isinstance(cx_test, cp.ndarray):
                cx_test = cp.asarray(cx_test)

            for i in range(x_test.shape[0]):
                test_points = cp.full(cx_train.shape[0], cx_test[i])
                distances = euclidean_dist(test_points, cx_train)

                k_indices = cp.argsort(distances)[:k]

                k_nearest_labels = [y_train[i] for i in k_indices]

                most_common = Counter(k_nearest_labels).most_common(1)
                predictions.append(most_common[0][0])

            return cp.array(predictions)

    @staticmethod
    def regression(x_train: np.ndarray | cp.ndarray,
                y_train: np.ndarray | cp.ndarray,
                x_test: np.ndarray | cp.ndarray,
                k: int,
                device: str = "cpu") -> np.ndarray | cp.ndarray:
        '''
        Predict the target value for the test data using K-nearest neighbors.
        :param x_train: Training data features.
        :param y_train: Training target values.
        :param x_test: Test data features.
        :param k: Number of neighbors to consider.
        :param device: CPU or GPU device.
        :return: Predicted target values for the test data.
        '''
        predictions = []

        if device == "cpu":
            nx_train = x_train
            ny_train = y_train
            nx_test = x_test

            if isinstance(nx_train, cp.ndarray):
                nx_train = cp.asnumpy(nx_train)

            if isinstance(ny_train, cp.ndarray):
                ny_train = cp.asnumpy(ny_train)

            if isinstance(nx_test, cp.ndarray):
                nx_test = cp.asnumpy(nx_test)

            for i in range(nx_test.shape[0]):
                test_points = np.full(nx_train.shape[0], nx_test[i])
                distances = euclidean_dist(test_points, nx_train)

                k_indices = np.argsort(distances)[:k]

                mean_prediction = np.mean([ny_train[i] for i in k_indices])
                predictions.append(mean_prediction)

            return np.array(predictions)
        else:
            cx_train = x_train
            cy_train = y_train
            cx_test = x_test

            if isinstance(cx_train, np.ndarray):
                cx_train = cp.asarray(cx_train)

            if isinstance(cy_train, np.ndarray):
                cy_train = cp.asarray(cy_train)

            if isinstance(cx_test, cp.ndarray):
                cx_test = cp.asarray(cx_test)

            for i in range(x_test.shape[0]):
                test_points = cp.full(cx_train.shape[0], cx_test[i])
                distances = euclidean_dist(test_points, cx_train)

                k_indices = cp.argsort(distances)[:k]

                mean_prediction = cp.mean(cp.array([cy_train[i] for i in k_indices]))
                predictions.append(mean_prediction)

            return cp.array(predictions)

    @staticmethod
    def find_optimal_k(x: np.ndarray | cp.ndarray,
                       y: list,
                       max_k: int) -> int:
        '''
        Find the optimal K value for K-nearest neighbors using cross-validation.
        :param x: Data features.
        :param y: Data labels.
        :param max_k: Maximum K value to test.
        :return: Best K value based on accuracy.
        '''
        best_k = 1
        best_accuracy = 0

        for k in range(1, max_k + 1):
            x_train, x_test, y_train, y_test = train_test_split(x, np.array(y), test_size=0.3, random_state=42)
            y_pred = KNearestNeighbor.predict(x_train, y_train, x_test, k)
            acc = accuracy(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_k = k

        return best_k