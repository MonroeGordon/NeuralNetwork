## Tools Module
Module containing various tools to aid in the training, testing, and/or deployment of machine learning and AI algorithms and models.

### Compression (compression.py)
Provides tools for data and image compression via various algorithms.

Details:
- _SVD Image Compression_: Compress image using singular value decompostion.

### Data Processing (data.py)
Provides tools for data processing.

Details:
- _Class Proportions_: Returns the proportion of the total dataset that each class value takes up.
- _Class Values_: Returns the unique class values in the dataset.
- _PCA Data Reconstruction_: Reconstructs a data matrix from a data matrix with dimensions reduced by PCA and its eigenvectors.
- _Train Test Split_: Splits a dataset into training and test sets.

### Datasets (datasets.py)
Provides tools for accessing a library of datasets that can be used for training and testing machine learning/AI models.

Details:
- **Data**: Class providing storage and access to input feature data and target value data from a dataset.
  - _Class Names_: Returns the names of the classes in the dataset.
  - _Feature Names_: Returns the names of the features in the dataset.
  - _Features_: Returns the input feature matrix.
  - _Targets_: Returns the target value matrix, which may contain class name index values or be a one hot encoding of the classes, depending on how the dataset was loaded.
- **Datasets**: Class providing access to a library of datasets and the ability to load them into a Data class for accessing and using the data.
  - _Load Iris_: Loads the Iris Flower dataset into a data class.
  - _Create Regression_: Creates a synthetic linear regression dataset.

### Metrics (metrics.py)
Provides tools for measuring various metrics in machine learning and AI models.

Details:
- _Accuracy_: Measures the accuracy of predictions.

### Data Plotting (plot.py)
Provides tools for plotting various machine learning and AI model data.

Details:
- _Plot OLS_: Plots ordinary least squares (OLS) linear regression prediction.

### Dimensionality Reduction (reduction.py)
Provides tools for dimensionality reduction via various algorithms.

Details:
- _LDA Projection_: Reduces the dimensionality of a matrix using a linear discriminant analysis (LDA) matrix projection.
- _PCA Dimensionality Reduction_: Reduces the dimensionaluty of a matrix using principal component analysis (PCA).
- _SVD Dimensionality Reduction_: Reduces the dimensionality of a matrix using singular value decomposition (SVD).
