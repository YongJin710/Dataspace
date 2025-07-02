import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Given data
temps = np.array([20, 30, 40, 50, 60], dtype=float)
viscosity = np.array([1.002, 0.797, 0.653, 0.547, 0.467], dtype=float)

# Lagrange Interpolation Function
def lagrange_interp(x_vals, y_vals, x):
    total = 0
    n = len(x_vals)
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        total += term
    return total

# Newton's Divided Difference Interpolation Functions
def newton_divided_diff(x_vals, y_vals):
    n = len(x_vals)
    coef = np.copy(y_vals)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x_vals[j:n] - x_vals[0:n-j])
    return coef

def newton_interp(x_vals, coef, x):
    n = len(coef)
    result = coef[-1]
    for i in range(n-2, -1, -1):
        result = result * (x - x_vals[i]) + coef[i]
    return result

# Interpolation targets
targets = [35, 45]
results = {'Lagrange': [], 'Newton': []}
timings = {'Lagrange': [], 'Newton': []}

# Interpolation calculations
newton_coef = newton_divided_diff(temps, viscosity)

for x in targets:
    # Lagrange
    start = time.perf_counter()
    y_lagrange = lagrange_interp(temps, viscosity, x)
    timings['Lagrange'].append(time.perf_counter() - start)
    results['Lagrange'].append(y_lagrange)

    # Newton
    start = time.perf_counter()
    y_newton = newton_interp(temps, newton_coef, x)
    timings['Newton'].append(time.perf_counter() - start)
    results['Newton'].append(y_newton)

# Tabulate results
summary = []
for i, temp in enumerate(targets):
    summary.append({
        "Temperature (°C)": temp,
        "Lagrange Viscosity (Pa.s)": round(results["Lagrange"][i], 6),
        "Lagrange Time (s)": f"{timings['Lagrange'][i]:.8f}",
        "Newton Viscosity (Pa.s)": round(results["Newton"][i], 6),
        "Newton Time (s)": f"{timings['Newton'][i]:.8f}"
    })

df = pd.DataFrame(summary)
print("\nInterpolation Results and Timing:\n")
print(df.to_string(index=False))

# Plotting
x_dense = np.linspace(20, 60, 400)
y_lagrange_dense = [lagrange_interp(temps, viscosity, x) for x in x_dense]
y_newton_dense = [newton_interp(temps, newton_coef, x) for x in x_dense]

plt.figure(figsize=(10, 6))
plt.plot(temps, viscosity, 'ko', label='Lab Data Points')
plt.plot(x_dense, y_lagrange_dense, 'r--', linewidth=2, label='Lagrange Interpolation')
plt.plot(x_dense, y_newton_dense, 'b-', linewidth=1.5, label='Newton Interpolation')
plt.scatter(targets, results['Lagrange'], color='red', marker='x', s=100, label='Lagrange Estimates')
plt.scatter(targets, results['Newton'], color='blue', marker='x', s=100, label='Newton Estimates')
plt.title('Viscosity Interpolation using Lagrange and Newton Methods')
plt.xlabel('Temperature (°C)')
plt.ylabel('Viscosity (Pa.s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
