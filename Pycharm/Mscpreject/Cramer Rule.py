import numpy as np
import time
import matplotlib.pyplot as plt


def cramers_rule(A, b):
    """Solve system Ax = b using Cramer's Rule."""
    n = A.shape[0]
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("Singular matrix: Cramer's Rule not applicable.")

    x = np.zeros(n)
    for i in range(n):
        A_i = np.copy(A)
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    return x


sizes = list(range(2, 11))
times = []

for n in sizes:
    while True:
        A = np.random.rand(n, n) * 10
        if abs(np.linalg.det(A)) > 1e-6:  # Avoid near-singular matrices
            break
    b = np.random.rand(n)

    start_time = time.perf_counter()
    x = cramers_rule(A, b)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    print(f"Size {n}x{n} - Time: {elapsed_time:.8f} seconds")

# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(sizes, times, marker='o', linestyle='-', color='blue')
plt.title("Execution Time of Cramer's Rule vs. System Size")
plt.xlabel("System Size (n x n)")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.show()

