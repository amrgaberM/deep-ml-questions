"""
Q19: Principal Component Analysis (PCA) Implementation
Source: Deep-ML Linear Algebra Collection
Description:
    Implement PCA (Principal Component Analysis) from scratch.
    The algorithm performs standardization, covariance computation,
    eigen decomposition, and selection of top-k principal components.
"""

import numpy as np 

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) on a dataset.

    Parameters
    ----------
    data : numpy.ndarray
        Input data matrix of shape (n_samples, n_features).
    k : int
        Number of principal components to extract.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n_features, k) containing the top-k
        principal components, rounded to 4 decimal places.
    """
    # Step 1: Standardize data
    X = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Step 2: Covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvalues/eigenvectors (descending order)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top-k principal components
    principal_components = eigenvectors[:, :k]
    
    # Step 6: Enforce sign convention (largest absolute entry is positive)
    for i in range(principal_components.shape[1]):
        if principal_components[np.argmax(np.abs(principal_components[:, i])), i] < 0:
            principal_components[:, i] *= -1
    
    return np.round(principal_components, 4)
