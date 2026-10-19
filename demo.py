"""Eval-lock card for real public circle-packing results (~10 s).

Inputs (data/public_constructions.json): AlphaEvolve B.12 n=26 / n=32 and B.13 n=21 (DeepMind's notebook,
Apache-2.0) and Aemon's n=26 packing as shown on aemon.ai. Read-only: the numbers are copied, nothing is called.
python demo.py
"""
import json

from evallock.packing import certify, check_exact, check_float
from evallock.slack import multipliers, reoptimize

D = json.load(open("data/public_constructions.json"))
SHINKA = {"relaxed_1e-6": "2.635983099011548", "post-processed": "2.6359828390115476", "exact_replication": "2.63597770931127"}
card = {}

print("EXACT CHECK (rational arithmetic, zero tolerance)")
for k, v in D.items():
    if k == "source_notes":
        continue
    r = check_exact(v["circles"], v["kind"])
    line = {"valid": r["valid"], "sum": r["sum_str"], "min_pair_gap": r["min_pair_gap"], "min_wall_gap": r["min_wall_gap"]}
    if v["kind"] == "square":
        m = multipliers(v["circles"])
        line.update(kkt_residual=round(m["kkt_residual"], 3), slack_value_1e6=1e-6 * m["sum_lambda"])
    card[k] = line
    print(f"  {k:<22} valid={r['valid']}  sum {r['sum_str']}  min gaps {r['min_pair_gap']:.1e} / {r['min_wall_gap']:.1e}"
          + (f"  KKT residual {line['kkt_residual']}" if "kkt_residual" in line else ""))

a = D["aemon_square_26"]["circles"]
print("\nWHAT A TOLERANT VERIFIER ADDS to the n=26 packing above (a search allowed to use the slack):")
card["slack_on_aemon_26"] = {}
for t in (1e-9, 1e-7, 1e-6):
    ro = reoptimize(a, t)
    c = certify(ro["circles"].tolist())
    card["slack_on_aemon_26"][f"{t:g}"] = {"tolerant_score": repr(ro["sum"]), "accepted_tol": check_float(ro["circles"], "square", t),
                                         "accepted_exact": check_exact(ro["circles"].tolist())["valid"], "certified": c["certified_str"]}
    print(f"  tol {t:g}: tolerant verifier accepts {ro['sum']:.10f} (+{ro['gain']:.2e}); exact: rejected; certified {c['certified_str'][:12]}")
print(f"  (ShinkaEvolve paper, PUBLIC: {SHINKA['relaxed_1e-6']} under 1e-6 slack -> exact replication {SHINKA['exact_replication']})")
json.dump(card, open("out/public_audit.json", "w"), indent=1, default=str)
print("\nwrote out/public_audit.json")
