"""
Q10: Orthogonal Projection of a Vector
Source: Deep-ML Linear Algebra Collection
Description:
    Project a vector v orthogonally onto a line defined by another vector L.
"""

import numpy as np

def orthogonal_projection(v: list[int | float], L: list[int | float]) -> list[float]:
    """
    Compute the orthogonal projection of vector v onto a line defined by vector L.

    Parameters
    ----------
    v : list[int | float]
        Input vector to be projected.
    L : list[int | float]
        Vector defining the line.

    Returns
    -------
    list[float]
        Projection of v onto L, with elements rounded to 3 decimal places.

    Raises
    ------
    ValueError
        If the line vector L is the zero vector.
    """
    v = np.array(v, dtype=float)
    L = np.array(L, dtype=float)
    
    if np.all(L == 0):
        raise ValueError("Line vector L cannot be the zero vector.")
    
    scalar = np.dot(v, L) / np.dot(L, L)
    proj = scalar * L
    return [round(x, 3) for x in proj]
