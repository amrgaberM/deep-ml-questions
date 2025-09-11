"""
Q9: CSR (Compressed Row Sparse) Matrix Conversion
Source: Deep-ML Linear Algebra Collection
Description:
    Convert a dense matrix into its Compressed Row Sparse (CSR) representation.
    The CSR format uses three arrays:
        - values: nonzero entries of the matrix
        - col_indices: column indices of the nonzero entries
        - row_ptr: index pointers to the start of each row
"""

import numpy as np

def compressed_row_sparse_matrix(dense_matrix: list[list[int | float]]):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    Parameters
    ----------
    dense_matrix : list[list[int | float]]
        A 2D list representing the dense matrix.

    Returns
    -------
    tuple[list[int | float], list[int], list[int]]
        A tuple containing:
        - values: list of nonzero entries
        - col_indices: list of column indices for each nonzero entry
        - row_ptr: list of row pointers indicating start of each row
    """
    A = np.array(dense_matrix)

    # Get nonzero positions
    rows, cols = np.nonzero(A)

    # Values and their column indices
    values = A[rows, cols].tolist()
    col_indices = cols.tolist()

    # Row pointer: cumulative count of nonzero elements per row
    counts = np.bincount(rows, minlength=A.shape[0])
    row_ptr = np.concatenate(([0], np.cumsum(counts))).tolist()

    return values, col_indices, row_ptr
