from linalg.eigen import eigen

import cupy as cp
import numpy as np

class LDA:
    '''
    Linear Discriminant Analysis (LDA) class. Provides algorithms for classification and dimensionality reduction using
    linear discriminant analysis.
    '''

    @staticmethod
    def between_class_scatter(x: np.ndarray | cp.ndarray,
                              y: np.ndarray | cp.ndarray,
                              means: dict,
                              device: str = "cpu") -> np.ndarray | cp.ndarray:
        '''
        Compute the between-class scatter matrix.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param means: Mean vectors for each class.
        :param device: CPU or GPU device.
        :return: Between-class scatter matrix.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ between_class_scatter: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 1:
            raise ValueError("@ lda_projection: parameter 'y' must be a vector (1-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("@ lda_projection: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        if device == "cpu":
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            overall_mean = np.mean(nx, axis=0)
            sb = np.zeros((nx.shape[1], nx.shape[1]))

            for c, mean in means.items():
                nc = np.sum(ny == c)
                mean = mean.reshape(-1, 1)
                overall_mean = overall_mean.reshape(-1, 1)
                sb += nc * (mean - overall_mean).dot((mean - overall_mean).T)
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            overall_mean = cp.mean(cx, axis=0)
            sb = cp.zeros((cx.shape[1], cx.shape[1]))

            for c, mean in means.items():
                nc = cp.sum(cy == c)
                mean = mean.reshape(-1, 1)
                overall_mean = overall_mean.reshape(-1, 1)
                sb += nc * (mean - overall_mean).dot((mean - overall_mean).T)

        return sb

    @staticmethod
    def fit(x: np.ndarray | cp.ndarray,
            y: np.ndarray | cp.ndarray,
            device: str="cpu") -> tuple:
        '''
        Fit the LDA model by computing the within-class and between-class scatter matrices.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param device: CPU or GPU device.
        :return: Projection matrix and class means.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ fit: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 1:
            raise ValueError("@ lda_projection: parameter 'y' must be a vector (1-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("@ lda_projection: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        means = LDA.means(x, y, device)
        sw = LDA.within_class_scatter(x, y, means, device)
        sb = LDA.between_class_scatter(x, y, means, device)

        e_vals, e_vecs = eigen(np.linalg.inv(sw).dot(sb), device=device)

        sorted_indices = np.argsort(e_vals)[::-1] if device == "cpu" else cp.argsort(e_vals)[::-1]

        w = e_vecs[:, sorted_indices[:len(means.keys()) - 1]]

        return w, means

    @staticmethod
    def means(x: np.ndarray | cp.ndarray,
              y: np.ndarray | cp.ndarray,
              device: str="cpu") -> dict:
        '''
        Compute the mean vectors for each class.
        :param x: Input feature matrix (number samples, number features)
        :param y: Class labels (number samples)
        :param device: CPU or GPU device.
        :return: Dictionary of mean vectors for each class.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ means: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 1:
            raise ValueError("@ lda_projection: parameter 'y' must be a vector (1-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("@ lda_projection: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        if device == "cpu":
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            classes = np.unique(ny)
            means = {}

            for c in classes:
                means[c] = np.mean(nx[ny == c], axis=0)

            return means
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            classes = cp.unique(cy)
            means = {}

            for c in classes:
                means[float(c)] = cp.mean(cx[cy == c], axis=0)

            return means

    @staticmethod
    def predict(x_transformed: np.ndarray | cp.ndarray,
                means: dict,
                device: str="cpu") -> list:
        '''
        Predict class labels based on transformed features using nearest mean classification.
        :param x_transformed: Transformed data (number samples, number classes - 1).
        :param means: Dictionary containing class means.
        :param device: CPU or GPU device.
        :return: Predicted class labels.
        '''
        preds = []

        if device == "cpu":
            nx = x_transformed

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            for x in nx:
                distances = {c: np.linalg.norm(x - mean) for c, mean in means.items()}
                preds.append(min(distances, key=distances.get))
        else:
            cx = x_transformed

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            for x in cx:
                distances = {c: cp.linalg.norm(x - mean) for c, mean in means.items()}
                preds.append(min(distances, key=distances.get))

        return preds

    @staticmethod
    def transform(x: np.ndarray | cp.ndarray,
                  w: np.ndarray | cp.ndarray,
                  device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Apply linear discriminant analysis (LDA) transformation to the input feature matrix.
        :param x: Input feature matrix (number samples, number features).
        :param w: Projection matrix obtained from fit.
        :param device: CPU or GPU device.
        :return: Transformed data (number samples, number classes - 1).
        '''
        if device == "cpu":
            nx = x
            nw = w

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(nw, cp.ndarray):
                nw = cp.asnumpy(nw)

            return np.dot(nx, nw)
        else:
            cx = x
            cw = w

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cw, np.ndarray):
                cw = cp.asarray(cw)

            return cp.dot(cx, cw)

    @staticmethod
    def within_class_scatter(x: np.ndarray | cp.ndarray,
                             y: np.ndarray | cp.ndarray,
                             means: dict,
                             device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Compute the within-class scatter matrix.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param means: Mean vectors for each class.
        :param device: CPU or GPU device.
        :return: Within-class scatter matrix.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ within_class_scatter: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if y.ndim != 1:
            raise ValueError("@ lda_projection: parameter 'y' must be a vector (1-dimensional ndarray).")

        if y.shape[0] != x.shape[0]:
            raise ValueError("@ lda_projection: parameter 'y' shape[0] must equal parameter 'x' shape[0].")

        if device == "cpu":
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            sw = np.zeros((nx.shape[1], nx.shape[1]))

            for c, mean in means.items():
                class_scatter = np.zeros((nx.shape[1], nx.shape[1]))

                for sample in nx[ny == c]:
                    sample = sample.reshape(-1, 1)
                    mean = mean.reshape(-1, 1)
                    class_scatter += (sample - mean).dot((sample - mean).T)

                sw += class_scatter
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            sw = cp.zeros((cx.shape[1], cx.shape[1]))

            for c, mean in means.items():
                class_scatter = cp.zeros((cx.shape[1], cx.shape[1]))

                for sample in cx[cy == c]:
                    sample = sample.reshape(-1, 1)
                    mean = mean.reshape(-1, 1)
                    class_scatter += (sample - mean).dot((sample - mean).T)

                sw += class_scatter

        return sw