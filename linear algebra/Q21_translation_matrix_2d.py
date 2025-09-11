import numpy as np

def translate_object(points, tx, ty):
    """
    Apply 2D translation to a set of points using homogeneous coordinates.

    Args:
        points (list[list[float]]): List of [x, y] points to be translated.
        tx (float): Translation along the x-axis.
        ty (float): Translation along the y-axis.

    Returns:
        list[list[float]]: Translated points as a list of [x, y].
    """
    # Translation matrix (3x3)
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]])
    
    # Convert points to homogeneous coordinates
    points_h = np.array([[x, y, 1] for x, y in points], dtype=float).T  # shape (3, N)
    
    # Apply transformation
    transformed = T @ points_h
    
    # Convert back to list of [x, y]
    return transformed[:2].T.tolist()
