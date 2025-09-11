"""
Q4: Dot Product Calculator
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a reusable function to calculate the dot product of two vectors.
"""

import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
    """
    Calculate the dot product of two vectors.

    Parameters
    ----------
    vec1 : list of int/float or numpy.ndarray
        First input vector.
    vec2 : list of int/float or numpy.ndarray
        Second input vector.

    Returns
    -------
    float
        Dot product of the two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return v1.dot(v2)
