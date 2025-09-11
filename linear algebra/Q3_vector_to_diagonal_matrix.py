"""
Q3: Vector to Diagonal Matrix
Source: Deep-ML Linear Algebra Collection
Description:
    Convert a given vector into a diagonal matrix.
"""

import numpy as np

def vector_to_diagonal_matrix(vec):
    """
    Convert a vector into a diagonal matrix.

    Parameters
    ----------
    vec : list of int/float or numpy.ndarray
        Input vector.

    Returns
    -------
    numpy.ndarray
        Diagonal matrix with elements of the vector on its main diagonal.
    """
    v = np.array(vec)
    return np.diag(v)
