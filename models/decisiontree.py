import cupy as cp
import numpy as np

class TreeNode:
    '''
    Tree Node class. Creates a tree node for a decision tree.
    '''

    def __init__(self,
                 feature: int=None,
                 threshold: float=None,
                 left=None,
                 right=None,
                 value=None):
        '''
        Initializes a tree node in a decision tree.
        :param feature: Input feature index.
        :param threshold: Split threshold.
        :param left: Left child tree node.
        :param right: Right child tree node.
        :param value: Tree node value.
        '''
        self._feature = feature
        self._threshold = threshold
        self._left = left
        self._right = right
        self._value = value

    def is_leaf(self) -> bool:
        '''
        Returns if this tree node is a leaf node.
        :return: True if this is a leaf node.
        '''
        return self._value is not None

    @property
    def feature(self) -> int | None:
        return self._feature

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def value(self):
        return self._value

class DecisionTree:
    '''
    Decision Tree class. Allows for creating a decision tree model for classification or regression tasks.
    '''

    def __init__(self,
                 max_depth: int=100,
                 min_samples_split: int=2,
                 device: str="cpu"):
        '''
        Initializes a decision tree.
        :param max_depth: Maximum depth of the decision tree.
        :param min_samples_split: Minimum samples per split.
        :param device: CPU or GPU device.
        '''
        if max_depth < 1:
            raise ValueError("DecisionTree: parameter 'max_depth' must exceed 0.")

        if min_samples_split < 1:
            raise ValueError("DecisionTree: parameter 'min_samples_split' must exceed 0.")

        if device != 'cpu' and device != 'gpu':
            raise ValueError("DecisionTree: parameter 'device' must be 'cpu' or 'gpu'.")

        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._device = device
        self._root = None
        self._n_class_labels = 0
        self._n_samples = 0
        self._n_features = 0

    def _best_split(self,
                    x: np.ndarray | cp.ndarray,
                    y: np.ndarray | cp.ndarray,
                    features) -> tuple:
        '''
        Finds the best tree node to split based on their information gain.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param features: List of input features.
        :return: Tuple containing the feature and threshold of the best split.
        '''
        split = {
            'score': -1,
            'feat': None,
            'threshold': None
        }

        if self._device == 'cpu':
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            for f in features:
                x_feat = nx[:, f]
                thresholds = np.unique(x_feat)

                for t in thresholds:
                    score = self._information_gain(x_feat, ny, t)

                    if score > split['score']:
                        split['score'] = score
                        split['feature'] = f
                        split['threshold'] = t
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            for f in features:
                x_feat = cx[:, f]
                thresholds = cp.unique(x_feat)

                for t in thresholds:
                    score = self._information_gain(x_feat, cy, t)

                    if score > split['score']:
                        split['score'] = score
                        split['feature'] = f
                        split['threshold'] = t

        return split['feature'], split['threshold']

    def _build_tree(self,
                    x: np.ndarray | cp.ndarray,
                    y: np.ndarray | cp.ndarray,
                    depth: int=0):
        '''
        Builds the decision tree from the inputs features and true values.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param depth: Current tree depth.
        :return: New tree node.
        '''
        if self._device == 'cpu':
            nx = x
            ny = y

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            self._n_samples, self._n_features = nx.shape
            self._n_class_labels = len(np.unique(ny))

            if self._is_finished(depth):
                most_common_label = np.argmax(np.bincount(ny.astype(np.int64)))
                return TreeNode(value=most_common_label)

            rnd_features = np.random.choice(self._n_features, self._n_features, replace=False)
            best_feature, best_threshold = self._best_split(nx, ny, rnd_features)

            left_idx, right_idx = self._create_split(nx[:, best_feature], best_threshold)
            left_child = self._build_tree(nx[left_idx, :], ny[left_idx], depth + 1)
            right_child = self._build_tree(nx[right_idx, :], ny[right_idx], depth + 1)
        else:
            cx = x
            cy = y

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            self._n_samples, self._n_features = cx.shape
            self._n_class_labels = len(cp.unique(cy))

            if self._is_finished(depth):
                most_common_label = cp.argmax(cp.bincount(cy.astype(cp.int64)))
                return TreeNode(value=most_common_label)

            rnd_features = cp.random.choice(self._n_features, self._n_features, replace=False)
            best_feature, best_threshold = self._best_split(cx, cy, rnd_features)

            left_idx, right_idx = self._create_split(cx[:, best_feature], best_threshold)
            left_child = self._build_tree(cx[left_idx, :], cy[left_idx], depth + 1)
            right_child = self._build_tree(cx[right_idx, :], cy[right_idx], depth + 1)

        return TreeNode(best_feature, best_threshold, left_child, right_child)

    def _create_split(self,
                      x: np.ndarray | cp.ndarray,
                      threshold: float) -> tuple:
        '''
        Creates a tree node split.
        :param x: Input feature matrix (number samples, number features).
        :param threshold: Split threshold.
        :return: Tuple containing indices of input features in the left split and right split.
        '''
        if self._device == 'cpu':
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            left_idx = np.argwhere(nx <= threshold).flatten()
            right_idx = np.argwhere(nx > threshold).flatten()
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            left_idx = cp.argwhere(cx <= threshold).flatten()
            right_idx = cp.argwhere(cx > threshold).flatten()

        return left_idx, right_idx

    def _entropy(self, y) -> float:
        '''
        Calculates the entropy of the class labels.
        :param y: Class labels (number samples).
        :return: Entropy value.
        '''
        if self._device == 'cpu':
            ny = y

            if isinstance(ny, cp.ndarray):
                ny = cp.asnumpy(ny)

            proportions = np.bincount(ny.astype(np.int64)) / len(ny)
            entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        else:
            cy = y

            if isinstance(cy, np.ndarray):
                cy = cp.asarray(cy)

            proportions = cp.bincount(cy.astype(cp.int64)) / len(cy)
            entropy = -cp.sum(cp.array([p * cp.log2(p) for p in proportions if p > 0]))

        return entropy

    def _information_gain(self,
                      x: np.ndarray | cp.ndarray,
                      y: np.ndarray | cp.ndarray,
                      threshold: float) -> float:
        '''
        Calculates the information gain between a parent tree node and its child tree nodes.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param threshold: Split threshold.
        :return: Information gain value.
        '''
        parent_loss = self._entropy(x)
        left_idx, right_idx = self._create_split(x, threshold)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])

        return parent_loss - child_loss

    def _is_finished(self, depth: int) -> bool:
        '''
        Returns if there are fewer samples than the minimum samples required for a split or if maximum depth is reached.
        :param depth: Current tree depth.
        :return: True if there are too few samples for a split or if maximum depth is reached.
        '''
        return depth >= self._max_depth or self._n_class_labels == 1 or self._n_samples < self._min_samples_split


    def _traverse_tree(self,
                       x: np.ndarray | cp.ndarray,
                       node: TreeNode):
        '''
        Traverse the decision tree.
        :param x: Input feature matrix (number samples, number features).
        :param node: Current tree node.
        :return: Next tree node value in the traversal.
        '''
        if node.is_leaf():
            return  node.value

        if ((self._device == 'cpu' and x[node.feature] <= node.threshold)
                or (self._device == 'gpu' and (cp.asarray(x))[node.feature] <= node.threshold)):
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def fit(self,
            x: np.ndarray | cp.ndarray,
            y: np.ndarray | cp.ndarray):
        '''
        Train a decision tree on the input features and true values.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        '''
        self._root = self._build_tree(x, y)

    def predict(self, x: np.ndarray | cp.ndarray):
        '''
        Makes a prediction from the decision tree based on the input features.
        :param x: Input feature matrix (number samples, number features).
        :return: Predictions from the decision tree.
        '''
        predictions = [self._traverse_tree(_x, self._root) for _x in x]

        return np.array(predictions) if self._device == 'cpu' else cp.array(predictions)