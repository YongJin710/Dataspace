import numpy as np

def naive_gaussian_elimination(A, b):
    n = len(b)
    # Augmented matrix
    M = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))

    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] = M[j, i:] - factor * M[i, i:]

    print("Upper Triangular Matrix [A|b]:")
    print(np.round(M, 3))

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if M[i, i] == 0:
            if M[i, -1] != 0:
                raise ValueError("System is inconsistent — no solution.")
            else:
                raise ValueError("System has infinite solutions.")
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]

    return x

# Define A and b
A = np.array([[2, 3, 1],
              [4, 1, 2],
              [-2, 5, -1]], dtype=float)
b = np.array([1, 2, 0], dtype=float)

try:
    solution = naive_gaussian_elimination(A, b)
    print("Solution vector x:", solution)
except ValueError as e:
    print("Error:", e)




