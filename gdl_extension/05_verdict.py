"""
05_verdict.py
════════════════════════════════════════════════════════════════════════
GDL extension, step 5 -- apply the success criterion written in README.md before any model was fitted.

For every GNN setting (gcn / gat x graph x target) on the primary sample:
  (i)  its out-of-sample loss must beat BOTH the best non-graph learned model and the best graph-as-features
       (SLX) model, with a country-block bootstrap 90% interval for the difference that excludes zero
       (the point estimate must also hold for the per-seed mean, i.e. without the benefit of seed ensembling);
  (ii) its gain over the no-graph twin (MLP) must shrink to at most a quarter when the graph is replaced by
       node-permuted placebo graphs.
Mean forecast: RMSE of the demeaned, clipped growth. Ceiling: pinball loss at 0.90 of the same target.
Anything else is a null result. Also reported: whether ANY learned model beats the training mean.

Usage: python 05_verdict.py [primary|nohom]
════════════════════════════════════════════════════════════════════════
"""
import json
import sys

import numpy as np
import pandas as pd

import gdl_common as G

SAMPLE = next((a for a in sys.argv[1:] if a in ("primary", "nohom")), "primary")
base = pd.read_csv(G.OUT / f"preds_baselines_{SAMPLE}.csv")
gnn = pd.read_csv(G.OUT / f"preds_gnn_{SAMPLE}.csv")
meta_cols = ["iso3", "year", "target_year", "target", "target_clip", "fold"]
df = base.merge(gnn.drop(columns=["target", "target_clip", "fold", "target_year"]), on=["iso3", "year"], how="inner")
assert len(df) == len(base) == len(gnn), "baselines and GNN must be scored on the same rows"
yt, yr, ct = df["target_clip"].to_numpy(), df["target_year"].to_numpy(), df["iso3"].to_numpy()
with open(G.OUT / f"gnn_{SAMPLE}.json", encoding="utf-8") as f:
    seeds = json.load(f)["results"]
GRAPHS = sorted({c.split("|")[1] for c in df.columns if c.startswith("gcn|")})


def loss_vec(col, ceiling):
    p = df[col].to_numpy()
    return G.pinball(p, yt, yr) if ceiling else G.squared_errors(p, yt, yr)


def score(col, ceiling):
    v = loss_vec(col, ceiling)
    return float(v.mean()) if ceiling else float(np.sqrt(v.mean()))


report = {"sample": SAMPLE, "n_test_rows": int(len(df)), "targets": {}}
for target, ceiling in (("mean", False), ("ceiling", True)):
    k_nograph = ["gbmq|none", "linq|none"] if ceiling else ["ridge|none", "rf|none", "gbm|none"]
    k_slx = [f"{k}|{g}" for g in GRAPHS for k in (["gbmq", "linq"] if ceiling else ["ridge", "rf", "gbm"])]
    twin = f"mlp|none|{target}"
    nograph = k_nograph + [twin]
    best_ng = min(nograph, key=lambda c: score(c, ceiling))
    best_slx = min(k_slx, key=lambda c: score(c, ceiling))
    trivial = {"training mean": score("mean", ceiling)} if not ceiling else {}
    sec = {"target": target, "metric": "pinball_dm" if ceiling else "rmse_dm", "best_non_graph": best_ng,
           "best_non_graph_score": score(best_ng, ceiling), "best_slx": best_slx, "best_slx_score": score(best_slx, ceiling),
           "twin_mlp_score": score(twin, ceiling), "trivial": trivial, "gnn": {}}
    if not ceiling:
        sec["learned_models_beating_training_mean"] = [c for c in nograph + k_slx + [c for c in df.columns if c.split("|")[0] in ("gcn", "gat")
                                                       and c.endswith("|mean")] if score(c, False) < score("mean", False)]
    for fam in ("gcn", "gat"):
        for g in GRAPHS:
            col = f"{fam}|{g}|{target}"
            s = score(col, ceiling)
            per_seed = seeds[col]["per_seed_mean"]
            vs = {}
            for tag, ref in (("non_graph", best_ng), ("slx", best_slx)):
                diff, lo, hi = G.block_bootstrap_diff(loss_vec(col, ceiling), loss_vec(ref, ceiling), ct, root=not ceiling)
                vs[tag] = {"reference": ref, "diff": diff, "boot90": [lo, hi]}
            plc = [score(c, ceiling) for c in df.columns if c.startswith(f"{col}|placebo")]
            real_gain = sec["twin_mlp_score"] - per_seed
            plc_gain = sec["twin_mlp_score"] - float(np.mean(plc)) if plc else None
            beats = all(per_seed < sec[k] for k in ("best_non_graph_score", "best_slx_score"))
            signif = all(v["boot90"][1] < 0 for v in vs.values())
            shrinks = bool(plc) and real_gain > 0 and plc_gain <= 0.25 * real_gain
            sec["gnn"][col] = {"ensemble_score": s, "per_seed_mean_score": per_seed, "vs": vs, "n_placebo": len(plc),
                               "placebo_mean_score": float(np.mean(plc)) if plc else None, "real_gain_over_mlp": real_gain,
                               "placebo_gain_over_mlp": plc_gain,
                               "criterion_i_beats_both_and_ci_excludes_zero": bool(beats and signif),
                               "criterion_ii_placebo_keeps_at_most_quarter": bool(shrinks),
                               "success": bool(beats and signif and shrinks)}
    report["targets"][target] = sec

for target, sec in report["targets"].items():
    unit = "pinball" if sec["target"] == "ceiling" else "RMSE"
    print(f"\n=== {target.upper()} ({unit}, demeaned) ===")
    print(f"best non-graph: {sec['best_non_graph']} {sec['best_non_graph_score']:.4f} | best SLX: {sec['best_slx']} {sec['best_slx_score']:.4f} | "
          f"MLP twin {sec['twin_mlp_score']:.4f}" + (f" | training mean {sec['trivial']['training mean']:.4f}" if sec["trivial"] else ""))
    if "learned_models_beating_training_mean" in sec:
        print(f"learned models beating the training mean: {sec['learned_models_beating_training_mean'] or 'none'}")
    for col, r in sec["gnn"].items():
        print(f"  {col:<24} ens {r['ensemble_score']:.4f} seed-mean {r['per_seed_mean_score']:.4f} | "
              f"vs non-graph {r['vs']['non_graph']['diff']:+.4f} [{r['vs']['non_graph']['boot90'][0]:+.4f},{r['vs']['non_graph']['boot90'][1]:+.4f}] | "
              f"vs SLX {r['vs']['slx']['diff']:+.4f} | placebo {r['placebo_mean_score']:.4f} | success {r['success']}")
n_success = sum(r["success"] for s in report["targets"].values() for r in s["gnn"].values())
report["n_successful_settings"] = n_success
report["verdict"] = ("at least one GNN setting meets the pre-registered criterion" if n_success
                     else "null result: no GNN setting meets the pre-registered criterion")
print("\nVERDICT:", report["verdict"])
json.dump(report, open(G.OUT / f"verdict_{SAMPLE}.json", "w"), indent=2, default=float)
print(f"saved outputs/verdict_{SAMPLE}.json")
