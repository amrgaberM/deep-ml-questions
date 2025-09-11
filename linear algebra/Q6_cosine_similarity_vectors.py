"""
Q6: Cosine Similarity
Source: Deep-ML Linear Algebra Collection
Description:
    Implement a function to calculate the cosine similarity between two vectors.
    Cosine similarity measures the cosine of the angle between vectors, giving
    a similarity score in the range [-1, 1].
"""

import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray
        First input vector.
    v2 : numpy.ndarray
        Second input vector.

    Returns
    -------
    float
        Cosine similarity between v1 and v2, rounded to 3 decimal places.

    Raises
    ------
    ValueError
        If either input vector is empty or has zero magnitude.
    """
    # Ensure input vectors are not empty
    if v1.size == 0 or v2.size == 0:
        raise ValueError("Input vectors cannot be empty.")
    
    # Compute dot product
    dot_product = np.dot(v1, v2)
    
    # Compute magnitudes
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Ensure no zero magnitude
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Input vectors must have non-zero magnitude.")
    
    # Cosine similarity formula
    similarity = dot_product / (norm_v1 * norm_v2)
    
    return round(similarity, 3)
