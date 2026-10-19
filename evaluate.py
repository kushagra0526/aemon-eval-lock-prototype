"""Held-out evaluation for both ways an eval can let "better" be faked.

python evaluate.py dev | heldout     (heldout seeds frozen in heldout/seeds.json before the first run)

A. Verifier slack (circle packing in the unit square, n = 10..30): a strict local optimum from random starts,
   then a search allowed to use a tolerance t. Measures the gain a tolerant verifier accepts, the KKT
   prediction t * sum(lambda), the n * t bound, and whether exact certification removes it.
B. Adaptive overfitting (ranker-style eval, 600 items, 400-step hill climb): naive vs locked holdout vs lock.
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

from evallock.packing import certify, check_exact, check_float
from evallock.selection import run_world
from evallock.slack import multipliers, random_optimum, reoptimize

TOLS = (1e-9, 1e-7, 1e-6)


def part_a(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 31))
    opt = random_optimum(n, rng)
    base = check_exact(opt.tolist())
    cert0 = certify(opt.tolist())
    lam = multipliers(opt)
    rows = []
    for t in TOLS:
        ro = reoptimize(opt, t)
        cert = certify(ro["circles"].tolist())
        rows.append({"t": t, "gain": ro["gain"], "pred": 0.99 * t * lam["sum_lambda"], "bound": n * t,
                     "passes_tolerant": check_float(ro["circles"], "square", t),
                     "passes_exact": check_exact(ro["circles"].tolist())["valid"],
                     "certified_minus_base": float(cert["certified_sum"] - cert0["certified_sum"])})
    return {"n": n, "kkt_residual": lam["kkt_residual"], "base_valid": base["valid"], "rows": rows}


def part_b(seed):
    return run_world(seed, signal=seed % 2 == 0)


def summarise(A, B):
    out = {"A_instances": len(A), "A_kkt_residual_median": float(np.median([a["kkt_residual"] for a in A]))}
    for i, t in enumerate(TOLS):
        r = [a["rows"][i] for a in A]
        ratio = [x["gain"] / x["pred"] for x in r if x["pred"] > 0]
        out[f"A_t={t:g}"] = {
            "median_gain": float(np.median([x["gain"] for x in r])),
            "gain_over_kkt_prediction_median": round(float(np.median(ratio)), 3),
            "within_20pct_of_prediction": f"{sum(abs(q - 1) <= .2 for q in ratio)}/{len(ratio)}",
            "over_n_t_bound": sum(x["gain"] > x["bound"] for x in r),
            "accepted_by_tolerant_verifier": sum(x["passes_tolerant"] for x in r),
            "accepted_by_exact": sum(x["passes_exact"] for x in r),
            "certified_gain_max": float(max(x["certified_minus_base"] for x in r)),
        }
    for sig in (False, True):
        r = [b for b in B if b["signal"] == sig]
        mean = lambda k: round(100 * float(np.mean([b[k] for b in r])), 2)
        key = "B_real_improvement_possible" if sig else "B_no_real_improvement"
        out[key] = {
            "worlds": len(r),
            "naive_claimed_gain_pts": mean("naive_claim"), "naive_true_gain_pts": mean("naive_true"),
            "holdout_claimed_gain_pts": mean("holdout_claim"), "holdout_true_gain_pts": mean("holdout_true"),
            "claims_win": {"naive": sum(b["naive_claims_win"] for b in r), "holdout": sum(b["holdout_claims_win"] for b in r),
                           "lock": sum(b["lock_claims_win"] for b in r)},
            "false_win (claimed, true gain <= 0)": {
                "naive": sum(b["naive_claims_win"] and b["naive_true"] <= 0 for b in r),
                "holdout": sum(b["holdout_claims_win"] and b["holdout_true"] <= 0 for b in r),
                "lock": sum(b["lock_claims_win"] and b["holdout_true"] <= 0 for b in r)},
        }
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split == "dev":
        sa, sb = list(range(20)), list(range(100))
    else:
        s = json.load(open("heldout/seeds.json"))
        sa, sb = s["packing"], s["selection"]
    with Pool() as p:
        A = p.map(part_a, sa)
        B = p.map(part_b, sb)
    res = summarise(A, B)
    print(json.dumps(res, indent=2, default=int))
    json.dump(res, open(f"out/{split}.json", "w"), indent=2, default=int)
