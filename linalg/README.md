## Linalg Module
Module containing linear algebra functions used to process matrices, systems of equations, and other AI relevant mathematics.

### Matrix Decomposition (decompostion.py)
Contains functions for decomposing matrices such as the LU decomposition and SVD. Decompositions of matrices allow for processing systems of linear equations and performing task such as dimensionality reduction, image compression, collaborative filtering, or principal component analysis.

Details:
- _Eignedecomposition_: Decomposes a square matrix into its eigenvalues and eigenvectors.
- _LU Decomposition_: Decomposes a square matrix into lower and upper triangular matrices.
- _Singular Value Decomposition (SVD)_: Decomposes a matrix into the product of orthogonal row/column basis vector matrices and a diagonal matrix of singular values.

### Eigenvalues and Eigenvectors (eigen.py)
Contains a function for calculating the eigenvalues and eigenvectors of a square matrix.

### System of Linear Equations (linear.py)
Contains functions for solving systems of linear equations with various methods.

Details:
- _Gaussian Elimination_: Solves a system of linear equations by performing row operations of the augmented matrix (matrix containing the coefficient matrix and the constant vector) to transform it into row-echelon form, allowing for the determination of unknown variables.
- _LU Solver_: Solves a system of linear equations by performing LU decomposition on the coefficient matrix, and then performing forward and backward substitutions to solve for Ly = b and Ux = y respe
- _Matrix Inversion_: Solves a system of linear equations by inverting the coefficient matrix (if possible) and multiplying it with both sides of Ax = b, giving x = A^-1 * b.

### Linear Regression (linreg.py)
Contains functions for performing linear regression using ordinary least squares (OLS).

Details:
- _OLS_: Calculates the coefficients of the linear regression model using ordinary least squars (OLS).
- _OLS Predict_: Makes predictions using the ordinary least squares (OLS) coefficients.

### Quadratic Equations (quadratic.py)
Contains functions for solving quadratic equations, performing discriminant analysis, and converting quadratic equations to vertex form.

### Miscellaneous Functions (functions.py)
Contains miscellaneous linear algebra functions that are relevant to machine learning and AI.

Details:
- _Euclidean Distance_: Calculates the Euclidean distance between two points.
