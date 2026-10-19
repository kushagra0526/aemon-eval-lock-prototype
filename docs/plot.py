"""python docs/plot.py -> docs/result.png from out/heldout.json + out/public_audit.json"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("out/heldout.json"))
p = json.load(open("out/public_audit.json"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
names = ["AlphaEvolve\n(published)", "Aemon\n(exact check)", "same packing + what a\n1e-6 verifier accepts"]
vals = [float(p["alphaevolve_square_26"]["sum"]), float(p["aemon_square_26"]["sum"]),
        float(p["slack_on_aemon_26"]["1e-06"]["tolerant_score"])]
base = vals[0]
cols = ["#98a0ae", "#292e93", "#c0397a"]
b = ax[0].bar(names, [(v - base) * 1e6 for v in vals], color=cols)
for bar, v in zip(b, vals):
    ax[0].text(bar.get_x() + bar.get_width() / 2, (v - base) * 1e6 + 2, f"{v:.7f}", ha="center", fontsize=9)
ax[0].set_ylabel("sum of radii above AlphaEvolve's published, x1e-6")
ax[0].set_title("Circle packing n=26: exact arithmetic vs a 1e-6 verifier")
k0, k1 = "B_no_real_improvement", "B_real_improvement_possible"
labels = ["naive\n(report on searched eval)", "holdout\n(search A, report B)", "lock\n(holdout + 95% bound)"]
fw = [r[k0]["false_win (claimed, true gain <= 0)"][m] + r[k1]["false_win (claimed, true gain <= 0)"][m] for m in ("naive", "holdout", "lock")]
b2 = ax[1].bar(labels, fw, color=["#c0397a", "#c98500", "#292e93"])
for bar, v in zip(b2, fw):
    ax[1].text(bar.get_x() + bar.get_width() / 2, v + 2, str(v), ha="center", fontweight="bold")
n = r[k0]["worlds"] + r[k1]["worlds"]
ax[1].set_title(f'"Beats baseline" claimed but not true, {n} held-out eval worlds')
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/result.png", dpi=130)
