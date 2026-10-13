"""
06_figure.py -- one figure summarising the verdict: loss of every GNN setting, of its placebo graphs and of the
reference models (best non-graph, best graph-as-features, MLP twin, training mean).

Usage: python 06_figure.py [primary|nohom]
"""
import json
import sys

import matplotlib.pyplot as plt

import gdl_common as G

SAMPLE = next((a for a in sys.argv[1:] if a in ("primary", "nohom")), "primary")
v = json.load(open(G.OUT / f"verdict_{SAMPLE}.json", encoding="utf-8"))
BLUE, RED, GREY, AMBER = "#2563EB", "#DC2626", "#6B7280", "#D97706"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, target, title, unit in ((axes[0], "mean", "Mean forecast", "RMSE of the demeaned growth"),
                                (axes[1], "ceiling", "Growth ceiling (q = 0.90)", "pinball loss")):
    sec = v["targets"][target]
    rows = sorted(sec["gnn"].items(), key=lambda kv: kv[1]["per_seed_mean_score"])
    ys = range(len(rows))
    ax.scatter([r["per_seed_mean_score"] for _, r in rows], ys, color=BLUE, zorder=3, label="GNN, real graph (mean of seeds)")
    ax.scatter([r["placebo_mean_score"] for _, r in rows], ys, color=RED, marker="x", zorder=3, label="GNN, node-permuted placebo")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c.rsplit("|", 1)[0].replace("|", " ") for c, _ in rows], fontsize=8)
    ax.axvline(sec["best_non_graph_score"], color=GREY, ls="--", lw=1.2, label=f"best non-graph ({sec['best_non_graph'].split('|')[0]})")
    ax.axvline(sec["best_slx_score"], color=AMBER, ls="--", lw=1.2, label=f"best graph-as-features ({sec['best_slx'].replace('|', ' ')})")
    ax.axvline(sec["twin_mlp_score"], color="black", ls=":", lw=1.2, label="MLP twin (no graph)")
    if sec["trivial"]:
        ax.axvline(sec["trivial"]["training mean"], color="#059669", ls="-.", lw=1.2, label="training mean")
    ax.set_xlabel(unit)
    ax.set_title(f"{title}: {sum(r['success'] for _, r in rows)} of {len(rows)} settings meet the criterion", fontsize=10)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle(f"Do graphs help? Out-of-sample loss, rolling-origin 2012-2023, sample '{SAMPLE}'", fontsize=11)
fig.tight_layout()
out = G.OUT / f"gdl_verdict_{SAMPLE}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
