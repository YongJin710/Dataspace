def f(x):
    return x**3 - 4*x - 9

def bisection(a, b, tol=1e-3):
    print(f"{'Iter':<5} {'a':<10} {'b':<10} {'f(a)':<10} {'f(b)':<10} {'x_mid':<12} {'f(x_mid)':<12}")
    iter_count = 1
    while (b - a) / 2 > tol:
        fa = f(a)
        fb = f(b)
        x_mid = (a + b) / 2
        f_mid = f(x_mid)

        print(f"{iter_count:<5} {a:<10.5f} {b:<10.5f} {fa:<10.5f} {fb:<10.5f} {x_mid:<12.5f} {f_mid:<12.5f}")

        if fa * f_mid < 0:
            b = x_mid
        else:
            a = x_mid
        iter_count += 1

    return x_mid

root = bisection(2, 3)
print(f"\nApproximate root: {root:.5f}")

