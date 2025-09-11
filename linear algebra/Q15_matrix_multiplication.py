"""
Q15: Matrix Multiplication
Source: Deep-ML Linear Algebra Collection
Description:
    Multiply two matrices if their dimensions are compatible.
"""

import numpy as np

def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]] | int:
    """
    Multiply two matrices.

    Parameters
    ----------
    a : list[list[int | float]]
        Left-hand matrix.
    b : list[list[int | float]]
        Right-hand matrix.

    Returns
    -------
    list[list[int | float]] | int
        Product of the two matrices as a list of lists.
        Returns -1 if the matrices have incompatible dimensions.
    """
    A = np.array(a)
    B = np.array(b)

    if A.shape[1] != B.shape[0]:
        return -1
    
    C = A.dot(B)
    return C.tolist()
