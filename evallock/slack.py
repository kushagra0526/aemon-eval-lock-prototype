"""How much score a verifier's tolerance hands to a search loop (unit-square packing, max sum of radii).

At a local optimum, relaxing every constraint g_i >= 0 to g_i >= -t raises the best score by about
t * sum(lambda_i), the KKT multipliers. Shrinking every radius by t undoes any relaxation, so n * t is a
hard upper bound. `reoptimize` measures the real gain by letting SLSQP use the slack.
"""
import numpy as np
from scipy.optimize import minimize, nnls


def _pairs(n):
    return np.triu_indices(n, 1)


def cons(z, n):
    """g(z) >= 0 for z = [x1, y1, r1, x2, ...]: pair gaps dist - ri - rj, then 4 wall gaps per circle."""
    a = z.reshape(n, 3)
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    i, j = _pairs(n)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    return np.concatenate([d - r[i] - r[j], x - r, y - r, 1 - x - r, 1 - y - r])


def jac(z, n):
    a = z.reshape(n, 3)
    x, y = a[:, 0], a[:, 1]
    i, j = _pairs(n)
    m = len(i)
    J = np.zeros((m + 4 * n, 3 * n))
    d = np.maximum(np.hypot(x[i] - x[j], y[i] - y[j]), 1e-15)
    ux, uy = (x[i] - x[j]) / d, (y[i] - y[j]) / d
    k = np.arange(m)
    J[k, 3 * i], J[k, 3 * i + 1], J[k, 3 * i + 2] = ux, uy, -1
    J[k, 3 * j], J[k, 3 * j + 1], J[k, 3 * j + 2] = -ux, -uy, -1
    c = np.arange(n)
    for w, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        row = m + w * n + c
        J[row, 3 * c], J[row, 3 * c + 1], J[row, 3 * c + 2] = dx, dy, -1
    return J


def multipliers(circles, active_tol=1e-6):
    """KKT multipliers of the active constraints (NNLS on -grad f = J_active^T lambda)."""
    z = np.asarray(circles, float).ravel()
    n = len(z) // 3
    g, J = cons(z, n), jac(z, n)
    act = g < active_tol
    grad = np.zeros(3 * n)
    grad[2::3] = 1.0
    lam, resid = nnls(J[act].T, -grad)
    return {"active": int(act.sum()), "sum_lambda": float(lam.sum()), "kkt_residual": float(resid)}


def solve(z0, n, tol=0.0, maxiter=1000):
    """Maximise sum of radii subject to g >= -tol (SLSQP, analytic Jacobians)."""
    f = lambda z: -z[2::3].sum()
    gf = np.zeros(3 * n)
    gf[2::3] = -1.0
    res = minimize(f, z0, jac=lambda z: gf, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda z: cons(z, n) + tol, "jac": lambda z: jac(z, n)}],
                   bounds=[(0, 1)] * (3 * n), options={"maxiter": maxiter, "ftol": 1e-16})
    return res.x


def reoptimize(circles, tol):
    z = np.asarray(circles, float).ravel()
    n = len(z) // 3
    z2 = solve(z, n, 0.99 * tol)                     # 99%: stay inside the slack despite float error
    g = cons(z2, n)
    return {"sum": float(z2[2::3].sum()), "gain": float(z2[2::3].sum() - z[2::3].sum()),
            "max_violation": float(max(0.0, -g.min())), "circles": z2.reshape(n, 3)}


def random_optimum(n, rng, starts=3):
    """A strict (tol = 0) local optimum from random starts; stands in for a search loop's output."""
    best = None
    for _ in range(starts):
        z = np.column_stack([rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n), np.full(n, 0.01)]).ravel()
        z = solve(z, n)
        if cons(z, n).min() < -1e-9:
            continue
        if best is None or z[2::3].sum() > best[2::3].sum():
            best = z
    return best.reshape(n, 3)
