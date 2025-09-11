"""
Q1: Matrix-Vector Dot Product
Source: Deep-ML Linear Algebra Collection
Description:
    Compute the dot product between a matrix and a vector.
"""

import numpy as np

def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    """
    Compute the dot product of a matrix and a vector.

    Parameters
    ----------
    a : list of lists of int/float
        Input 2D matrix.
    b : list of int/float
        Input 1D vector.

    Returns
    -------
    list of int/float
        Result of the matrix-vector multiplication.
        Returns -1 if dimensions are invalid.
    """
    # Convert inputs to NumPy arrays
    A = np.array(a)
    B = np.array(b)

    # Validate dimensions
    if A.ndim != 2 or B.ndim != 1:
        return -1
    if A.shape[1] != B.shape[0]:
        return -1

    # Matrix-vector multiplication
    result = A.dot(B)
    
    # Convert back to list (to keep same output format as your original function)
    return result.tolist()
