import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# define a symbolic variable for angle
theta = sp.symbols('theta')

# define 2D rotation matrix
R = sp.Matrix([
    [sp.cos(theta), -sp.sin(theta)],
    [sp.sin(theta), sp.cos(theta)]
])

# compute and simplify R^T * R
orthogonality_check_1 = sp.simplify(R.T * R)
print("R^T * R (should be identity):")
sp.pprint(orthogonality_check_1)
print()

# compute and simplify R * R^T
orthogonality_check_2 = sp.simplify(R * R.T)
print("R * R^T (should also be identity):")
sp.pprint(orthogonality_check_2)
print()

# compute and simplify inverse of R (R \ I in MATLAB)
R_inv = R.inv()
inverse_check = sp.simplify(R.T - R_inv)
print("R^T - R_inv (should be zero matrix):")
sp.pprint(inverse_check)
print()

# substitute specific angle for demonstration (e.g., 45 degrees)
theta_val = np.pi / 4  # 45 degrees in radians
R_numeric = np.array(R.subs(theta, theta_val)).astype(np.float64)

# define vector to rotate
original_vector = np.array([1, 0])  # unit vector along x-axis
rotated_vector = R_numeric @ original_vector

# plot original and rotated vectors
plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

# plot original vector
plt.quiver(0, 0, original_vector[0], original_vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector')

# plot rotated vector
plt.quiver(0, 0, rotated_vector[0], rotated_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label='Rotated Vector')

# set plot limits, labels
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title('2D Rotation Matrix Demonstration')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()