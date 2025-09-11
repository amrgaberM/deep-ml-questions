import numpy as np
import math

def svd_2x2(A):
    """
    Compute SVD of a 2x2 matrix using eigenvalues and eigenvectors method.
    Returns U, S, V such that A = U @ diag(S) @ V
    
    Args:
        A: 2x2 numpy array or list
    
    Returns:
        tuple: (U, S, V) where A = U @ diag(S) @ V
    """
    # Convert to numpy array if needed
    A = np.array(A, dtype=float)
    
    # Step 1: Compute A^T*A and A*A^T
    AT = A.T
    ATA = AT @ A
    AAT = A @ AT
    
    # Step 2: Find eigenvalues using characteristic polynomial
    def get_eigenvalues(M):
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        
        # Coefficients of characteristic polynomial: λ² - trace*λ + det = 0
        trace = a + d
        det = a*d - b*c
        
        # Use quadratic formula
        discriminant = trace**2 - 4*det
        
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
        else:
            # This shouldn't happen for A^T*A or A*A^T (they're positive semidefinite)
            sqrt_disc = math.sqrt(-discriminant)
            lambda1 = trace / 2
            lambda2 = trace / 2
            
        return [max(lambda1, lambda2), min(lambda1, lambda2)]  # Sort in descending order
    
    def get_eigenvector(M, eigenval):
        """Get normalized eigenvector for given eigenvalue"""
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        
        # Solve (M - λI)v = 0
        # [(a-λ)  b  ] [x] = [0]
        # [c    (d-λ)] [y]   [0]
        
        # Try first row: (a-λ)x + by = 0
        coeff_x = a - eigenval
        coeff_y = b
        
        if abs(coeff_y) > 1e-10:
            # Solve for x in terms of y: x = -by/(a-λ)
            y = 1.0
            x = -coeff_y / coeff_x if abs(coeff_x) > 1e-10 else 0.0
        elif abs(coeff_x) > 1e-10:
            # b ≈ 0, so any y works, x = 0
            x = 0.0
            y = 1.0
        else:
            # Try second row: cx + (d-λ)y = 0
            coeff_x2 = c
            coeff_y2 = d - eigenval
            
            if abs(coeff_x2) > 1e-10:
                y = 1.0
                x = -coeff_y2 / coeff_x2 if abs(coeff_y2) > 1e-10 else 0.0
            else:
                # Default to standard basis
                x, y = 1.0, 0.0
        
        # Normalize
        v = np.array([x, y])
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        else:
            v = np.array([1.0, 0.0])
        
        return v
    
    # Step 3: Get eigenvalues (they're the same for both A^T*A and A*A^T)
    eigenvalues = get_eigenvalues(ATA)
    
    # Step 4: Get eigenvectors for A*A^T (columns of U)
    u1 = get_eigenvector(AAT, eigenvalues[0])
    u2 = get_eigenvector(AAT, eigenvalues[1])
    
    # Step 5: Get eigenvectors for A^T*A (columns of V^T)
    v1 = get_eigenvector(ATA, eigenvalues[0])
    v2 = get_eigenvector(ATA, eigenvalues[1])
    
    # Step 6: Construct matrices
    U = np.column_stack([u1, u2])
    VT = np.column_stack([v1, v2]).T  # We want V^T for the final result
    
    # Singular values are square roots of eigenvalues
    singular_values = [math.sqrt(max(0, lam)) for lam in eigenvalues]
    S = np.array(singular_values)
    
    # Step 7: Handle sign consistency for A = U @ diag(S) @ V
    # We need to make sure the signs work out correctly
    
    # Method: Use the relation A*v_i = s_i * u_i to determine correct signs
    for i in range(2):
        if S[i] > 1e-10:  # Non-zero singular value
            v_i = VT[i, :]  # i-th row of VT = i-th column of V
            expected_u = A @ v_i / S[i]  # What u_i should be
            actual_u = U[:, i]
            
            # If they point in opposite directions, flip one of them
            if np.dot(expected_u, actual_u) < 0:
                VT[i, :] = -VT[i, :]
    
    # Return the matrices such that A = U @ diag(S) @ VT
    return U, S, VT

