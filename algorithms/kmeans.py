import cupy as cp
import numpy as np

class KMeans:
    '''
    KMeans class. Provides the algorithm for K-means clustering.
    '''

    @staticmethod
    def cluster(x: np.ndarray | cp.ndarray,
                k: int=1,
                max_iters: int=100,
                random_state: int=0,
                device: str="cpu") -> tuple:
        '''
        Perform K-means clustering.
        :param x: Data points to be clustered (number data points, number features).
        :param k: Number of clusters.
        :param max_iters: Maximum number of iterations.
        :param random_state: Seed for reproducibility.
        :param device: CPU or GPU device.
        :return: Cluster centroids and labels for each point.
        '''
        np.random.seed(random_state)
        n, d = x.shape

        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            centroids = nx[np.random.choice(n, k, replace=False)]
            labels = None

            for i in range(max_iters):
                distances = np.linalg.norm(nx[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                new_centroids = np.array([nx[labels == i].mean(axis=0) for i in range(k)])

                if np.all(centroids == new_centroids):
                    break

                centroids = new_centroids
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            centroids = cx[cp.random.choice(n, k, replace=False)]
            labels = None

            for i in range(max_iters):
                distances = cp.linalg.norm(cx[:, cp.newaxis] - centroids, axis=2)
                labels = cp.argmin(distances, axis=1)

                new_centroids = cp.array([cx[labels == i].mean(axis=0) for i in range(k)])

                if cp.all(centroids == new_centroids):
                    break

                centroids = new_centroids

        return centroids, labels