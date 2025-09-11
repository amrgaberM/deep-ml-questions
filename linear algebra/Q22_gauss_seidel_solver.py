import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve a system of linear equations using the Gauss-Seidel iterative method.

    Args:
        A (list[list[float]] or np.ndarray): Coefficient matrix.
        b (list[float] or np.ndarray): Constant terms vector.
        n (int): Number of iterations to perform.
        x_ini (list[float], optional): Initial guess for the solution vector. Defaults to zeros.

    Returns:
        list[float]: Approximate solution vector after n iterations.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m = len(b)
    
    # Initial guess
    if x_ini is None:
        x = np.zeros(m)
    else:
        x = np.array(x_ini, dtype=float)
    
    # Iterative process
    for _ in range(n):
        for i in range(m):
            sum1 = np.dot(A[i, :i], x[:i])        # Use updated values
            sum2 = np.dot(A[i, i+1:], x[i+1:])    # Use old values
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
    
    return x.tolist()
