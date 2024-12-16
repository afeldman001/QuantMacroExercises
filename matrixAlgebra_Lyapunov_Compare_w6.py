import numpy as np
import time

# function: solve discrete Lyapunov equation using Kronecker product
def lyapunov_kron(A, SIGu):
    n = A.shape[0]
    I = np.eye(n)
    kron_matrix = np.kron(A, A)
    vec_SIGu = SIGu.flatten(order='F')
    vec_SIGy = np.linalg.solve(np.eye(n**2) - kron_matrix, vec_SIGu)
    return vec_SIGy.reshape(n, n, order='F')

# function: solve discrete Lyapunov equation using doubling algorithm
def lyapunov_doubling(A, SIGu, max_iter=500, tol=1e-25):
    n = A.shape[0]
    SIGy_old = np.eye(n)  # initial guess
    A_old = A
    SIGu_old = SIGu
    for _ in range(max_iter):
        SIGy_new = A_old @ SIGy_old @ A_old.T + SIGu_old
        diff = np.max(np.abs(SIGy_new - SIGy_old))
        if diff < tol:
            break
        SIGu_old = A_old @ SIGu_old @ A_old.T + SIGu_old
        A_old = A_old @ A_old
        SIGy_old = SIGy_new
    return SIGy_new

# small example
print("=== Small Example ===")
A = np.array([
    [0.5, 0, 0],
    [0.1, 0.1, 0.3],
    [0.0, 0.2, 0.3]
])
SIGu = np.array([
    [2.25, 0, 0],
    [0, 1, 0.5],
    [0, 0.5, 0.74]
])

# Kronecker Product Method
start = time.time()
SIGy_kron = lyapunov_kron(A, SIGu)
end = time.time()
print(f"Kronecker method time: {end - start:.6f} seconds")

# Doubling Algorithm
start = time.time()
SIGy_doubling = lyapunov_doubling(A, SIGu)
end = time.time()
print(f"Doubling method time: {end - start:.6f} seconds")

# compare results
difference = np.max(np.abs(SIGy_kron - SIGy_doubling))
print(f"Maximum absolute difference: {difference}")

# large example
print("\n=== Large Example ===")
n = 100

# generate random matrices with spectral radius < 1 for A
while True:
    SIGu = np.random.randn(n, n)
    SIGu = SIGu.T @ SIGu  # snsure SIGu is positive semi-definite
    Arand = np.random.randn(n, n)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(Arand)))
    A = Arand / (spectral_radius + 1e-5)  # normalize to ensure spectral radius < 1
    if np.max(np.abs(np.linalg.eigvals(A))) < 1:
        break

# Kronecker Product Method
start = time.time()
SIGy_kron = lyapunov_kron(A, SIGu)
end = time.time()
print(f"Kronecker method time (large): {end - start:.6f} seconds")

# Doubling Algorithm
start = time.time()
SIGy_doubling = lyapunov_doubling(A, SIGu)
end = time.time()
print(f"Doubling method time (large): {end - start:.6f} seconds")

# compare results
difference = np.max(np.abs(SIGy_kron - SIGy_doubling))
print(f"Maximum absolute difference (large): {difference}")

