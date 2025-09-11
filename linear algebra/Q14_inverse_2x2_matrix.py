"""
Q14: Inverse of a 2×2 Matrix
Source: Deep-ML Linear Algebra Collection
Description:
    Compute the inverse of a 2×2 matrix using NumPy.
"""

import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    """
    Calculate the inverse of a 2×2 matrix.

    Parameters
    ----------
    matrix : list[list[float]]
        A 2×2 matrix represented as a list of lists.

    Returns
    -------
    list[list[float]]
        The inverse of the input matrix as a list of lists.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular (not invertible).
    """
    m = np.array(matrix, dtype=float)
    mi = np.linalg.inv(m)
    return mi.tolist()
