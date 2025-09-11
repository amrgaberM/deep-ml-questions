import numpy as np

def transform_basis(B, C):
    """
    Computes the change of basis matrix P from basis B to basis C.

    The matrix P satisfies the equation: [v]_C = P * [v]_B,
    where [v]_B are the coordinates of a vector v in basis B, and
    [v]_C are the coordinates of the same vector v in basis C.

    The formula used is P = C⁻¹ * B.

    Args:
        B (list of lists or np.ndarray): A list of basis vectors for the starting basis B.
                                         Each inner list is a vector.
        C (list of lists or np.ndarray): A list of basis vectors for the target basis C.
                                         Each inner list is a vector.

    Returns:
        np.ndarray: The transformation matrix P that converts coordinates
                    from basis B to basis C.
    """
    # 1. Create matrices from the basis vectors.
    # By convention, basis vectors form the COLUMNS of the matrix.
    # We convert the list of row vectors to a NumPy array and then transpose it.
    B_matrix = np.array(B)
    C_matrix = np.array(C)
    # 2. Check if C is invertible.
    # The matrix C must be square and have a non-zero determinant for its
    # inverse to exist. A non-zero determinant means the vectors in C
    # are linearly independent and thus form a valid basis.
    if np.linalg.det(C_matrix) == 0:
        raise ValueError("The basis C is not invertible. Vectors may be linearly dependent.")

    # 3. Calculate the inverse of matrix C.
    C_inverse = np.linalg.inv(C_matrix)

    # 4. Compute the transformation matrix P = C⁻¹ * B
    # The '@' operator is NumPy's dedicated matrix multiplication operator.
    P_B_to_C = C_inverse @ B_matrix

    return P_B_to_C

