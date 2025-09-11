"""
Q16: Reduced Row Echelon Form (RREF)
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a function to compute the Reduced Row Echelon Form (RREF) of a matrix.
"""

import numpy as np

def rref(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Reduced Row Echelon Form (RREF) of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix.

    Returns
    -------
    numpy.ndarray
        Matrix in Reduced Row Echelon Form (RREF).
    """
    A = matrix.astype(float)  # work with floats to avoid integer division issues
    rows, cols = A.shape
    r = 0  # row tracker
    
    for c in range(cols):
        if r >= rows:
            break
        
        # Step 1: Find the pivot (max abs value in column c, from row r downwards)
        pivot = np.argmax(np.abs(A[r:rows, c])) + r
        if np.isclose(A[pivot, c], 0):  # skip if column is zero
            continue
        
        # Step 2: Swap pivot row into current row
        A[[r, pivot]] = A[[pivot, r]]
        
        # Step 3: Normalize pivot row (make pivot = 1)
        A[r] = A[r] / A[r, c]
        
        # Step 4: Eliminate other rows in column c
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, c] * A[r]
        
        r += 1  # move to next row
    
    return A
