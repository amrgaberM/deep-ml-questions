"""
Q5: Column Space (Image) Basis
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a function to compute a basis for the column space (image)
    of a given matrix using Gaussian elimination to identify pivot columns.
"""

import numpy as np

def matrix_image(A: np.ndarray) -> np.ndarray:
    """
    Return a basis for the column space (image) of matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        Input matrix.

    Returns
    -------
    numpy.ndarray
        A matrix whose columns form a basis for the column space of A.
    """
    A = A.astype(float)  # ensure float for safe division
    m, n = A.shape
    B = A.copy()
    
    pivot_cols = []
    row = 0
    
    # Gaussian elimination to Row Echelon Form
    for col in range(n):
        # Find pivot row
        pivot = np.argmax(np.abs(B[row:, col])) + row
        if np.isclose(B[pivot, col], 0):  # skip if no pivot in this column
            continue
        
        # Swap pivot row into position
        if pivot != row:
            B[[row, pivot]] = B[[pivot, row]]
        
        # Eliminate entries below pivot
        for r in range(row + 1, m):
            if B[r, col] != 0:
                factor = B[r, col] / B[row, col]
                B[r] -= factor * B[row]
        
        pivot_cols.append(col)  # store pivot column index
        row += 1
        if row == m:
            break
    
    # Extract pivot columns from ORIGINAL matrix
    basis = A[:, pivot_cols]
    return basis
