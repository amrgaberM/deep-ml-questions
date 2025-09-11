import numpy as np

def svd_2x2_singular_values(A):
    """
    Compute SVD of a 2x2 matrix using direct eigenvalue computation.
    
    Args:
        A: NumPy array of shape (2, 2)
    
    Returns:
        Tuple (U, Σ, V_T) where:
        - U: 2x2 orthogonal matrix (left singular vectors)
        - Σ: length 2 array of singular values
        - V_T: 2x2 orthogonal matrix (transpose of right singular vectors)
    """
    A = np.array(A, dtype=float)
    
    # Step 1: Compute A^T A
    AtA = A.T @ A
    
    # Step 2: Find eigenvalues and eigenvectors of A^T A directly
    # For 2x2 symmetric matrix AtA = [[a, b], [b, c]]
    a = AtA[0, 0]  # a11
    b = AtA[0, 1]  # a12 = a21
    c = AtA[1, 1]  # a22
    
    # Solve characteristic equation: det(AtA - λI) = 0
    # λ² - (a + c)λ + (ac - b²) = 0
    trace = a + c
    det = a * c - b * b
    
    # Use quadratic formula
    discriminant = trace * trace - 4 * det
    if discriminant < 0:
        discriminant = 0  # Handle numerical errors
    
    sqrt_disc = np.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    
    # Ensure non-negative eigenvalues (handle numerical errors)
    lambda1 = max(0, lambda1)
    lambda2 = max(0, lambda2)
    
    # Step 3: Compute eigenvectors
    if abs(b) < 1e-15:  # Already diagonal
        if a >= c:
            V = np.array([[1, 0], [0, 1]])  # Identity
        else:
            V = np.array([[0, 1], [1, 0]])  # Swap
            lambda1, lambda2 = lambda2, lambda1
    else:
        # Eigenvector for lambda1
        if abs(a - lambda1) > abs(c - lambda1):
            # Use first row: (a - λ1)x + by = 0
            v1 = np.array([-b, a - lambda1])
        else:
            # Use second row: bx + (c - λ1)y = 0
            v1 = np.array([c - lambda1, -b])
        v1 = v1 / np.linalg.norm(v1)
        
        # Eigenvector for lambda2
        if abs(a - lambda2) > abs(c - lambda2):
            # Use first row: (a - λ2)x + by = 0
            v2 = np.array([-b, a - lambda2])
        else:
            # Use second row: bx + (c - λ2)y = 0
            v2 = np.array([c - lambda2, -b])
        v2 = v2 / np.linalg.norm(v2)
        
        V = np.column_stack([v1, v2])
    
    # Step 4: Compute singular values
    sigma1 = np.sqrt(lambda1)
    sigma2 = np.sqrt(lambda2)
    
    # Step 5: Ensure singular values are in descending order
    if sigma2 > sigma1:
        sigma1, sigma2 = sigma2, sigma1
        V = V[:, [1, 0]]  # Swap columns
    
    # Step 6: Compute U matrix using U = A * V * Σ^(-1)
    U = np.zeros((2, 2))
    
    # Compute columns of U
    if sigma1 > 1e-15:
        U[:, 0] = A @ V[:, 0] / sigma1
    else:
        U[:, 0] = np.array([1.0, 0.0])
    
    if sigma2 > 1e-15:
        U[:, 1] = A @ V[:, 1] / sigma2
    else:
        # Make orthogonal to first column
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
    
    # Step 7: Orthogonalize U using Gram-Schmidt
    U[:, 1] = U[:, 1] - np.dot(U[:, 1], U[:, 0]) * U[:, 0]
    norm_u2 = np.linalg.norm(U[:, 1])
    if norm_u2 > 1e-15:
        U[:, 1] = U[:, 1] / norm_u2
    
    # Step 8: Fix signs to match expected output
    # First, ensure the first column of U has positive first element if possible
    if U[0, 0] < 0:
        U[:, 0] = -U[:, 0]
        V[:, 0] = -V[:, 0]
    
    # For the second column, check the SVD relationship A*v = σ*u
    if sigma2 > 1e-15:
        expected_u2 = A @ V[:, 1] / sigma2
        # If current u2 and expected u2 point in opposite directions, flip both u2 and v2
        if np.dot(U[:, 1], expected_u2) < 0:
            U[:, 1] = -U[:, 1]
            V[:, 1] = -V[:, 1]
    
    # Step 9: Ensure proper rotation matrices (det = +1)
    if np.linalg.det(U) < 0:
        U[:, 1] = -U[:, 1]
        V[:, 1] = -V[:, 1]
    
    if np.linalg.det(V) < 0:
        V[:, 1] = -V[:, 1]
        U[:, 1] = -U[:, 1]
    
    # Return results
    Sigma = np.array([sigma1, sigma2])
    V_T = V.T
    
    return U, Sigma, V_T

