"""
Q8: Scalar Multiplication of a Matrix
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a function to multiply a matrix by a scalar value.
"""

import numpy as np

def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    """
    Multiply a matrix by a scalar.

    Parameters
    ----------
    matrix : list[list[int | float]]
        Input matrix as a list of lists.
    scalar : int | float
        Scalar value to multiply with.

    Returns
    -------
    list[list[int | float]]
        Matrix after scalar multiplication, as a list of lists.
    """
    m = np.array(matrix)
    result = m * scalar   # element-wise scalar multiplication
    return result.tolist()
