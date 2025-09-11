"""
Q18: Jacobi Linear Solver
Source: Deep-ML Linear Algebra Collection
Description:
    Solve a system of linear equations Ax = b using the Jacobi iterative method.
"""

import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list[float]:
    """
    Solve a linear system using the Jacobi iterative method.

    Parameters
    ----------
    A : numpy.ndarray
        Coefficient matrix (must be square and diagonally dominant for convergence).
    b : numpy.ndarray
        Right-hand side vector.
    n : int
        Number of iterations to perform.

    Returns
    -------
    list[float]
        Approximate solution vector after n iterations, rounded to 4 decimal places.
    """
    A, b = np.array(A, dtype=float), np.array(b, dtype=float)
    m = len(b)
    
    # Initial guess: zero vector
    x = np.zeros(m)
    
    for _ in range(n):
        x_new = np.zeros_like(x)
        for i in range(m):
            # Exclude diagonal element from sum
            s = sum(A[i][j] * x[j] for j in range(m) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        # Round after each iteration
        x = np.round(x_new, 4)
    
    return x.tolist()
