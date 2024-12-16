import sympy as sp
import numpy as np

# generate random dimensions between 1 and 10 for the matrices
dim = np.random.randint(1, 11, 4)  # generate 4 random integers between 1 and 10
print(f"Randomly chosen dimensions: {dim}")

# create symbolic matrices using random generated dimensions
D = sp.MatrixSymbol('D', dim[0], dim[1])
E = sp.MatrixSymbol('E', dim[1], dim[2])
F = sp.MatrixSymbol('F', dim[2], dim[3])

# matrix product D * E * F
DEF = D * E * F

# vectorize result
vecDEF = sp.Matrix(DEF).vec()

# vectorization of E
vecE = sp.Matrix(E).vec()

# Kronecker product
F_transpose = sp.Matrix(F).transpose()
D_matrix = sp.Matrix(D)
kron_matrix = sp.Matrix(sp.kronecker_product(F_transpose, D_matrix))
kron_result = kron_matrix * vecE

# fully expand both expressions
vecDEF_expanded = sp.expand(vecDEF)
kron_result_expanded = sp.expand(kron_result)

# print intermediate values to compare (uncomment if you want to see the matrices)
#print("\nvec(D*E*F):\n", vecDEF_expanded)
#print("\nFully expanded kron(F',D)*vec(E):\n", kron_result_expanded)

# validate equality of expanded expressions
if sp.simplify(vecDEF_expanded - kron_result_expanded) == sp.zeros(vecDEF.shape[0], vecDEF.shape[1]):
    print("\nExpanded expressions are identical")
else:
    print("\nExpanded expressions are not identical")
