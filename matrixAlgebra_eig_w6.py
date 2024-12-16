import numpy as np

# define matrix A
A = np.array([
    [0.5, 0, 0],
    [0.1, 0.1, 0.3],
    [0, 0.2, 0.3]
])

# compute eigenvalues of A
eigenvalues = np.linalg.eigvals(A)

# check if absolute values of eigenvalues < 1
eigenvalues_abs_less_than_1 = np.abs(eigenvalues) < 1

# display result
print(eigenvalues)
print(eigenvalues_abs_less_than_1)
