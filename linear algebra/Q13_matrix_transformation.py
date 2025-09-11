import numpy as np

def transform_matrix(A, T, S):
    A, T, S = np.array(A, dtype=float), np.array(T, dtype=float), np.array(S, dtype=float)
    
    # Check invertibility (det != 0)
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1
    
    try:
        T_inv = np.linalg.inv(T)
        result = T_inv @ A @ S   # Matrix multiplication
        return result.round(3).tolist()  # Round to 3 decimals
    except np.linalg.LinAlgError:
        return -1


