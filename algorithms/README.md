## Algorithms Module
Module containing machine learning and AI algorithms that are separate from models.

### K-Nearest Neighbors (knn.py)
Contains a class for predicting class labels or target values using the K-nearest neighbors algorithm.

Details:
- **KNN**: Class containing functions for predicting class labels or target values.
  - _Predict_: Predicts class labels for the test data after training on the training data.
  - _Regression_: Predicts target values for the test data after training on the training data.
  - _Find Optimal K_: Finds the best K value using cross-validation on the dataset.

### Linear Discriminant Analysis (lda.py)
Contains a class for performing linear discriminant analysis on an input feature matrix.

Details:
- **LDA**: Class containing functions for finding the mean vectors for each class, and for computing between-class and within-class scatter matrices.
  - _Between Class Scatter_: Computes the between-class scatter matrix of the input feature matrix and class labels, using the computed mean vectors for each class.
  - _Fit_: Fits the LDA model to the input feature matrix and class labels.
  - _Means_: Computes the mean vectors for each class.
  - _Predict_: Predicts class labels from transformed features using nearest mean classification.
  - _Transform_: Applies LDA transformation to the input feature matrix.
  - _Within Class Scatter_: Computes the within-class scatter matrix of the input feature matrix and class labels, using the computed mean vectors for each class.

### Principal Component Analysis (pca.py)
Contains a class for performing principal component analysis on an input feature matrix.

Details:
- **PCA**: Class containing a function for calculating the covariance matrix of an input feature matrix.
  - _Covariance Matrix_: Calculates the covariance matrix of an input feature matrix.
