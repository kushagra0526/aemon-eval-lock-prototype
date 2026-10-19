"""Exact verification and certified scores for circle-packing evals (AlphaEvolve B.12 / B.13 style).

Circles are rows (x, y, r). Inputs may be decimal strings (exact as published) or floats.
kind = "square": inside [0,1]^2.  kind = "rect4": minimum bounding rectangle has width + height <= 2.
"""
import itertools
from decimal import Decimal, getcontext
from fractions import Fraction as F

import numpy as np

getcontext().prec = 50


def exact(circles):
    return [[F(str(v)) if not isinstance(v, F) else v for v in row] for row in circles]


def _dec(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def pair_gaps(c):
    """Exact sign + 50-digit distance gap dist - (ri + rj) for every pair."""
    out = []
    for i, j in itertools.combinations(range(len(c)), 2):
        (x1, y1, r1), (x2, y2, r2) = c[i], c[j]
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        ok = d2 >= (r1 + r2) ** 2                     # exact rational comparison
        gap = _dec(d2).sqrt() - _dec(r1 + r2)
        out.append((i, j, ok, gap))
    return out


def wall_gaps(c, kind):
    if kind == "square":
        return [(i, min(x - r, y - r, 1 - x - r, 1 - y - r)) for i, (x, y, r) in enumerate(c)]
    w = max(x + r for x, _, r in c) - min(x - r for x, _, r in c)
    h = max(y + r for _, y, r in c) - min(y - r for _, y, r in c)
    return [(-1, 2 - (w + h))]                        # one perimeter constraint


def check_exact(circles, kind="square"):
    """Zero-tolerance check in exact arithmetic. Returns a dict a reader can audit."""
    c = exact(circles)
    pg = pair_gaps(c)
    wg = wall_gaps(c, kind)
    worst_pair = min(pg, key=lambda t: t[3])
    worst_wall = min(wg, key=lambda t: t[1])
    total = sum(r for _, _, r in c)
    return {
        "n": len(c), "kind": kind, "valid": all(t[2] for t in pg) and all(g >= 0 for _, g in wg),
        "overlaps": sum(not t[2] for t in pg), "wall_violations": sum(g < 0 for _, g in wg),
        "sum": total, "sum_str": str(_dec(total))[:18],
        "min_pair_gap": float(worst_pair[3]), "min_wall_gap": float(worst_wall[1]),
    }


def check_float(circles, kind="square", tol=0.0):
    """What a float verifier with absolute slack `tol` accepts (OpenEvolve-style uses 1e-6)."""
    a = np.asarray(circles, dtype=float)
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    i, j = np.triu_indices(len(a), 1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    if np.any(d < r[i] + r[j] - tol):
        return False
    if kind == "square":
        return bool(np.all(np.minimum.reduce([x - r, y - r, 1 - x - r, 1 - y - r]) >= -tol))
    w = (x + r).max() - (x - r).min()
    h = (y + r).max() - (y - r).min()
    return bool(w + h <= 2 + tol)


def certify(circles, kind="square"):
    """Largest uniform radius shrink delta that makes the packing exactly valid; certified score = sum - n*delta.

    Shrinking every radius by delta adds 2*delta to every pair gap, delta to every wall gap and 4*delta to the
    rect4 perimeter slack, so delta = max(pair violation / 2, wall violation, perimeter violation / 4).
    """
    c = exact(circles)
    viol = [F(0)]
    for _, _, ok, gap in pair_gaps(c):
        if not ok:
            viol.append(F(str(-gap)) / 2 + F(1, 10 ** 40))   # round the irrational gap up
    for _, g in wall_gaps(c, kind):
        if g < 0:
            viol.append(-g / (4 if kind == "rect4" else 1))
    delta = max(viol)
    shrunk = [[x, y, r - delta] for x, y, r in c]
    res = check_exact(shrunk, kind)
    assert res["valid"], "certification failed"
    return {"delta": float(delta), "certified_sum": res["sum"], "certified_str": res["sum_str"],
            "given_sum": sum(r for _, _, r in c)}
