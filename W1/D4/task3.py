import numpy as np

# 1. Create a 5x4 matrix of random integers between 0 and 100
np.random.seed(42)
matrix = np.random.randint(0, 101, size=(5, 4))

# 2. Compute the mean of each column (axis 0) and row (axis 1)
col_means = np.mean(matrix, axis=0)
row_means = np.mean(matrix, axis=1)
print(matrix)
print(col_means)
print(row_means)
# 3. Center the data by subtracting column means
centered_matrix = matrix - col_means
print(centered_matrix)
# 4. Find the index of the maximum value in the entire matrix
flat_max_idx = np.argmax(matrix)
wax= matrix.flat[flat_max_idx]
print(wax)
max_idx = np.unravel_index(flat_max_idx, matrix.shape)
print(max_idx)
# 5. Threshold matrix: 0 if < 50, 1 if >= 50
binary_matrix = np.where(matrix < 50, 0, 1)
print(binary_matrix)

# 6. Compute standard deviation of each row
row_stds = np.std(matrix, axis=1)