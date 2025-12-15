## Algorithms Module
Module containing machine learning and AI algorithms that are separate from models.

### Hidden Markov Model and Markov Chain (hmm.py)
Contains a class for processing a hidden Markov model or Markov chain that use observations to infer hidden states.

Details:
- **HMM**: Class containing functions for processing a hidden Markov model or Markoc chain.
  - _Estimate_: Estimates the parameters of a hidden Markov model using the Baum-Welch algorithm.
  - _Likely States_: Finds the most likely sequence of hidden states using the Viterbi algorithm.
  - _Sample_: Samples from a target distribution using the Metropolis-Hastings algorithm.
  - _Sequence Probability_: Calculates the probability of a sequence of observable symbols using the forward algorithm.
  - _Stationary Distribution_: Calculates the stationary distribution of a Markov chain.
  - _Transition Probability Matrix_: Creates a transition probability matrix for a Markov chain.

### K-Means Clustering (kmeans.py)
Contains a class for clustering data points into clusters using the K-means clustering algorithm.

Details:
- **KMeans**: Class containing a function for clustering data points using K-means clustering.
  - _Cluster_: Cluster data points into the specified number of clusters using K-means clustering.

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
 
### Particle Filter (particle.py)
Contains a class for processing a particle filter that helps find approximate solutions to filtering problems.

Details:
- **ParticleFilter**: Class containing functions for processing a particle filter.
  - _State Transition_: Processes the state transition function for the system.
  - _Observation_: Function for relating state to observations.
  - _Filter_: Performs the particle filtering using the state transition and observation functions.

### Principal Component Analysis (pca.py)
Contains a class for performing principal component analysis on an input feature matrix.

Details:
- **PCA**: Class containing a function for calculating the covariance matrix of an input feature matrix.
  - _Covariance Matrix_: Calculates the covariance matrix of an input feature matrix.
