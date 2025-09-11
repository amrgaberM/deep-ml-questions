import numpy as np

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    """
    Calculate the eigenvalues of a 2x2 matrix using NumPy.
    Returns eigenvalues sorted from highest to lowest.
    """
    A = np.array(matrix, dtype=float)

    # Ensure matrix is 2x2
    if A.shape != (2, 2):
        raise ValueError("Input must be a 2x2 matrix")

    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(A)

    # Sort from highest to lowest
    return sorted(eigenvalues.tolist(), reverse=True)
