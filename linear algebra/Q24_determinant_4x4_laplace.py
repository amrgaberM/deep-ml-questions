def determinant_4x4(matrix: list[list[int|float]]) -> float:
    """
    Calculate the determinant of a 4x4 matrix using Laplace's Expansion method.
    
    Args:
        matrix: A 4x4 matrix represented as a list of lists
        
    Returns:
        The determinant of the matrix as a float
    """
    
    def get_minor(mat, row, col):
        """
        Get the minor matrix by removing the specified row and column.
        
        Args:
            mat: The matrix to get minor from
            row: Row index to remove
            col: Column index to remove
            
        Returns:
            The minor matrix with specified row and column removed
        """
        return [[mat[i][j] for j in range(len(mat[i])) if j != col] 
                for i in range(len(mat)) if i != row]
    
    def determinant_recursive(mat):
        """
        Recursively calculate determinant using Laplace's expansion.
        
        Args:
            mat: Square matrix of any size (2x2, 3x3, or 4x4)
            
        Returns:
            The determinant of the matrix
        """
        n = len(mat)
        
        # Base case: 2x2 matrix
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        # Base case: 1x1 matrix (shouldn't occur in our use case)
        if n == 1:
            return mat[0][0]
        
        # Recursive case: expand along the first row
        det = 0
        for col in range(n):
            # Calculate cofactor: (-1)^(row+col) * minor_determinant
            # Since we're expanding along row 0, row+col = 0+col = col
            cofactor = ((-1) ** col) * mat[0][col]
            minor = get_minor(mat, 0, col)
            det += cofactor * determinant_recursive(minor)
        
        return det
    
    # Validate input
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Input must be a 4x4 matrix")
    
    return determinant_recursive(matrix)

