"""
Q2: Transpose of a Matrix
Source: Deep-ML Linear Algebra Collection
Description:
    Calculate the transpose of a given matrix.
"""

import numpy as np

def transpose_matrix(a):
    """
    Compute the transpose of a matrix.

    Parameters
    ----------
    a : list of lists of int/float
        Input 2D matrix.

    Returns
    -------
    numpy.ndarray
        Transposed matrix.
    """
    return np.array(a).T
