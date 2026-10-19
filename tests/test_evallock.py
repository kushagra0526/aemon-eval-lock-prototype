import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evallock.packing import certify, check_exact, check_float
from evallock.selection import paired_lower, run_world
from evallock.slack import multipliers, reoptimize

D = json.load(open(ROOT / "data" / "public_constructions.json"))


def test_tiny_overlap_passes_tolerant_but_not_exact():
    c = [["0.25", "0.5", "0.25"], ["0.75", "0.5", "0.2500000001"]]   # overlap 1e-10, and wall breach 1e-10
    assert check_float(c, "square", 1e-6)
    r = check_exact(c)
    assert not r["valid"] and r["overlaps"] == 1
    cert = certify(c)
    assert check_exact([[x, y, F(str(rr)) - F(cert["delta"])] for x, y, rr in c])["valid"]
    assert cert["certified_sum"] < F("0.5000000001")


def test_public_constructions_exact_and_alphaevolve_values():
    r = check_exact(D["alphaevolve_square_26"]["circles"])
    assert r["valid"] and r["sum_str"].startswith("2.63586275")
    assert check_exact(D["alphaevolve_rect4_21"]["circles"], "rect4")["valid"]
    assert check_exact(D["aemon_square_26"]["circles"])["sum_str"].startswith("2.635983082")


def test_kkt_predicts_slack_gain_and_bound_holds():
    a = D["aemon_square_26"]["circles"]
    m = multipliers(a)
    assert m["kkt_residual"] < 1e-6
    ro = reoptimize(a, 1e-6)
    assert abs(ro["gain"] / (0.99e-6 * m["sum_lambda"]) - 1) < 0.05
    assert ro["gain"] <= 26e-6
    assert check_float(ro["circles"], "square", 1e-6) and not check_exact(ro["circles"].tolist())["valid"]


def test_certify_removes_slack_gain():
    a = D["aemon_square_26"]["circles"]
    ro = reoptimize(a, 1e-7)
    assert certify(ro["circles"].tolist())["certified_sum"] < check_exact(a)["sum"]


def test_paired_lower_bound():
    rng = np.random.default_rng(0)
    base = (rng.random(400) < 0.5).astype(float)
    better = np.maximum(base, (rng.random(400) < 0.3).astype(float))
    assert paired_lower(better, base, rng) > 0
    assert paired_lower(base, base, rng) == 0


def test_naive_report_overstates_on_null_world():
    r = run_world(3, signal=False)
    assert r["naive_claim"] > r["naive_true"] + 0.02
