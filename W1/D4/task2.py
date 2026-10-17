import numpy as np

# Set seed for reproducible output
np.random.seed(42)

# 1. Create a 6x6 matrix of random integers between 1 and 50
matrix = np.random.randint(1, 51, size=(6, 6))

# 2. Extract the second row (index 1)
second_row = matrix[1, :]

# 3. Extract the last column (index -1)
last_column = matrix[:, -1]

# 4. Extract the 3x3 sub-matrix from rows 1–3 and cols 1–3 (indices 1:4, 1:4)
sub_matrix = matrix[1:4, 1:4]

# 5. Boolean mask for elements > 25
mask_gt_25 = matrix > 25
elements_gt_25 = matrix[mask_gt_25]

# 6. Replace all even numbers with -1
matrix_modified = matrix.copy()
matrix_modified[matrix_modified % 2 == 0] = -1

# 7. Count elements > 30
count_gt_30 = np.sum(matrix > 30)