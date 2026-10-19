"""Adaptive overfitting of a fixed eval set by a research loop, and the locked-holdout fix.

World: a ranker/classifier with parameters theta. Item i is solved iff a_i + b_i . theta + e_i > 0, so every
eval item has its own quirks (b_i, e_i) a search can latch onto. True quality = solve rate on fresh items.
Some worlds have a real improvement direction (mean of b_i != 0), some have none.

Search loop: (1+1) hill climb that keeps a change when the eval score goes up (what "learn from every
result" does to a fixed eval). Reporting rules compared:
  naive     best score on the eval it searched on, minus the baseline on the same eval
  holdout   search on half A, measure once on locked half B (paired with baseline)
  lock      holdout + claim "beats baseline" only if a paired one-sided 95% bound on B is > 0
"""
import numpy as np


def world(rng, signal):
    d = 20
    beta = np.zeros(d)
    if signal:
        beta[: rng.integers(1, 4)] = rng.uniform(0.05, 0.25)
    return {"d": d, "beta": beta, "mu": rng.uniform(-0.3, 0.6), "spread": rng.uniform(0.6, 1.2)}


def items(w, m, rng):
    return {"a": rng.normal(w["mu"], 1.0, m), "b": w["beta"] + rng.normal(0, w["spread"], (m, w["d"])),
            "e": rng.logistic(0, 0.3, m)}


def solved(it, theta):
    return (it["a"] + it["b"] @ theta + it["e"] > 0).astype(float)


def search(it, steps, rng, step=0.15):
    theta = np.zeros(it["b"].shape[1])
    best = solved(it, theta).mean()
    for _ in range(steps):
        cand = theta + rng.normal(0, step, len(theta))
        s = solved(it, cand).mean()
        if s > best:
            theta, best = cand, s
    return theta


def paired_lower(x_new, x_base, rng, boots=2000):
    """One-sided 95% lower bound of mean(x_new - x_base) by paired bootstrap."""
    d = x_new - x_base
    idx = rng.integers(0, len(d), (boots, len(d)))
    return float(np.quantile(d[idx].mean(1), 0.05))


def run_world(seed, signal, m=600, steps=400, fresh=40_000):
    rng = np.random.default_rng(seed)
    w = world(rng, signal)
    ev = items(w, m, rng)
    pop = items(w, fresh, rng)
    zero = np.zeros(w["d"])
    true_base = solved(pop, zero).mean()

    # naive: search and report on the same full eval set
    th = search(ev, steps, np.random.default_rng(seed + 1))
    naive_claim = solved(ev, th).mean() - solved(ev, zero).mean()
    naive_true = solved(pop, th).mean() - true_base

    # holdout: search on A, report once on locked B
    A = {k: v[: m // 2] for k, v in ev.items()}
    B = {k: v[m // 2:] for k, v in ev.items()}
    th2 = search(A, steps, np.random.default_rng(seed + 1))
    xb_new, xb_base = solved(B, th2), solved(B, zero)
    hold_claim = xb_new.mean() - xb_base.mean()
    hold_true = solved(pop, th2).mean() - true_base
    lower = paired_lower(xb_new, xb_base, np.random.default_rng(seed + 2))
    return {"signal": signal, "naive_claim": naive_claim, "naive_true": naive_true,
            "holdout_claim": hold_claim, "holdout_true": hold_true, "lock_claims_win": lower > 0,
            "naive_claims_win": naive_claim > 0, "holdout_claims_win": hold_claim > 0}
