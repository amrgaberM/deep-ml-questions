import numpy as np

def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    A = np.array(dense_matrix)

    # Nonzero positions
    rows, cols = np.nonzero(A)

    # Sort indices by column, then by row (CSC order)
    order = np.lexsort((rows, cols))
    rows, cols = rows[order], cols[order]

    # Values and their row indices
    values = A[rows, cols].tolist()
    row_ind = rows.tolist()

    # Column pointer: counts per column
    counts = np.bincount(cols, minlength=A.shape[1])
    col_ptr = np.concatenate(([0], np.cumsum(counts))).tolist()

    return values, row_ind, col_ptr
