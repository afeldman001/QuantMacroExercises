import numpy as np

# define covariance matrix SIGu
SIGu = np.array([
    [2.25, 0, 0],
    [0, 1, 0.5],
    [0, 0.5, 0.74]
])

# perform Cholesky decomposition (lower triangular matrix)
P = np.linalg.cholesky(SIGu)

# compute SIGe_sqrt (diagonal elements of P)
SIGe_sqrt = np.diag(P)

# construct SIGe (diagonal matrix of squared diagonal elements of P)
SIGe = np.diag(SIGe_sqrt**2)

# solve for W using matrix right division (equivalent to P / diag(SIGe_sqrt) in MATLAB solution)
W = P @ np.linalg.inv(np.diag(SIGe_sqrt))

# check if W * SIGe * W' equals SIGu
reconstructed_SIGu = W @ SIGe @ W.T
is_equal = np.allclose(reconstructed_SIGu, SIGu)

# print results
print("Original SIGu:")
print(SIGu)
print("\nCholesky decomposition P:")
print(P)
print("\nDiagonal elements of P (SIGe_sqrt):")
print(SIGe_sqrt)
print("\nSIGe (constructed diagonal matrix):")
print(SIGe)
print("\nMatrix W:")
print(W)
print("\nReconstructed SIGu from W * SIGe * W':")
print(reconstructed_SIGu)
print("\nDoes W * SIGe * W' equal SIGu?:", is_equal)
