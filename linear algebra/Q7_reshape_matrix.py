"""
Q7: Reshape Matrix
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a function to reshape a matrix into a new shape,
    if and only if the total number of elements matches.
"""

import numpy as np

def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    """
    Reshape a given matrix into a new specified shape.

    Parameters
    ----------
    a : list[list[int | float]]
        The input matrix as a list of lists.
    new_shape : tuple[int, int]
        The desired new shape (rows, columns).

    Returns
    -------
    list[list[int | float]]
        Reshaped matrix as a list of lists. Returns an empty list if
        the reshape is not possible due to mismatched element count.
    """
    A = np.array(a)
    if A.size != np.prod(new_shape):  # check if reshape is possible
        return []
    reshaped_matrix = A.reshape(new_shape)
    return reshaped_matrix.tolist()
