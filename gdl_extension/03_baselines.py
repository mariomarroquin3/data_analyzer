"""
03_baselines.py
════════════════════════════════════════════════════════════════════════
GDL extension, step 3 -- baselines the graph neural networks must beat.

  no graph          training mean, last-year growth, ridge, random forest, gradient boosting;
                    quantile versions for the growth ceiling (gradient boosting, linear quantile regression)
  graph as features the econometric analogue (SLX): the same learners with the weighted average of the
                    neighbours' features added, for four graphs (trade, inverse distance, 4 nearest, borders)
  placebo           the same SLX models with node-permuted graphs (structure kept, alignment with the
                    countries destroyed), a few draws per graph

Evaluation: rolling-origin blocks defined in gdl_common.py, hyper-parameters tuned only on the last
three target years of each training window. Primary metric: RMSE of the cross-sectionally demeaned
next-year growth (see README.md).

Usage:  python 03_baselines.py [primary|nohom] [--quick]
════════════════════════════════════════════════════════════════════════
"""
import json
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gdl_common as G

warnings.filterwarnings("ignore")
SAMPLE = next((a for a in sys.argv[1:] if a in ("primary", "nohom")), "primary")
QUICK = "--quick" in sys.argv
N_PLACEBO = 1 if QUICK else 5
FOLDS = G.FOLDS[-1:] if QUICK else G.FOLDS

d, FEATS = G.load_dataset(SAMPLE)
ISO = sorted(d["iso3"].unique())
GRAPHS = G.build_graphs(ISO)
print(f"sample={SAMPLE}: {len(ISO)} economies, {len(d)} rows, {len(FEATS)} features, graphs={list(GRAPHS)}")
y = d["target"].to_numpy(float)
yc = y.copy()                 # training target of the current fold: clipped to the training window's 1st-99th percentile range, then year-demeaned
years_t = d["target_year"].to_numpy()

# ── learners and fixed hyper-parameter grids ─────────────────────────
def make(kind, p):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=p["depth"], min_samples_leaf=p["leaf"], max_features=0.5,
                                     n_jobs=3, random_state=0)
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=p["depth"], subsample=0.8, random_state=0)
    if kind == "gbmq":
        return GradientBoostingRegressor(loss="quantile", alpha=G.QUANTILE, n_estimators=150, learning_rate=0.05,
                                         max_depth=p["depth"], subsample=0.8, random_state=0)
    if kind == "linq":
        return make_pipeline(StandardScaler(), QuantileRegressor(quantile=G.QUANTILE, alpha=p["alpha"], solver="highs"))
    raise ValueError(kind)


GRID = {
    "ridge": [{"alpha": a} for a in (1, 10, 100, 1000)],
    "rf": [{"depth": 6, "leaf": 5}, {"depth": None, "leaf": 10}],
    "gbm": [{"depth": 2}, {"depth": 3}],
    "gbmq": [{"depth": 2}, {"depth": 3}],
    "linq": [{"alpha": 0.0}, {"alpha": 0.0005}, {"alpha": 0.005}],
}
QUANTILE_KINDS = {"gbmq", "linq"}


def val_loss(kind, pred, idx):
    if kind in QUANTILE_KINDS:
        return float(np.mean(G.pinball(pred, yc[idx], years_t[idx])))
    return float(np.sqrt(np.mean(G.squared_errors(pred, yc[idx], years_t[idx]))))


def fit_predict(kind, X, fit, val, train_all, test):
    """Tune on the validation block (imputer fitted on the fit rows only), refit on the whole training window."""
    Xf, (Xv,) = G.impute(X.loc[fit], [X.loc[val]])
    best, best_loss = None, np.inf
    for p in GRID[kind]:
        m = make(kind, p).fit(Xf, yc[fit])
        loss = val_loss(kind, m.predict(Xv), val)
        if loss < best_loss:
            best, best_loss = p, loss
    Xt, (Xs,) = G.impute(X.loc[train_all], [X.loc[test]])
    return make(kind, best).fit(Xt, yc[train_all]).predict(Xs), best


# ── design matrices: no graph, SLX with real graphs, SLX with node-permuted graphs ──
base = d[FEATS].copy()
designs = {"none": base}
for gname, W in GRAPHS.items():
    designs[gname] = pd.concat([base, G.neighbour_features(d, FEATS, W, ISO)], axis=1)
placebo = {}
for gname, W in GRAPHS.items():
    for k in range(N_PLACEBO):
        rng = np.random.default_rng(1000 * (list(GRAPHS).index(gname) + 1) + k)
        placebo[(gname, k)] = pd.concat([base, G.neighbour_features(d, FEATS, G.permute_graph(W, rng), ISO)], axis=1)

settings = [("ridge", "none"), ("rf", "none"), ("gbm", "none"), ("gbmq", "none"), ("linq", "none")]
for gname in GRAPHS:
    settings += [(k, gname) for k in ("ridge", "rf", "gbm", "gbmq", "linq")]
for (gname, k) in placebo:
    settings += [(kd, f"{gname}|placebo{k}") for kd in ("ridge", "rf", "gbm", "gbmq")]

preds = {}
chosen = {}
t0 = time.time()
CACHE = G.OUT / f"baselines_cache_{SAMPLE}{'_quick' if QUICK else ''}.pkl"   # one entry per finished fold, so a run can resume
cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
rows_test, clip_vals, bounds = [], [], {}
for fold in FOLDS:
    fit, val, train_all, test = G.fold_indices(d, fold)
    rows_test.append(test)
    lo_b, hi_b = G.winsor_bounds(y[train_all])
    bounds[str(fold)] = [lo_b, hi_b]
    y_clip = np.clip(y, lo_b, hi_b)
    yc = G.demean_by_year(y_clip, years_t)       # models are trained on the clipped, year-demeaned target
    clip_vals.append(y_clip[test])
    if fold in cache:
        for nm, arr in cache[fold]["preds"].items():
            preds.setdefault(nm, {})[fold] = arr
        for nm, b in cache[fold]["chosen"].items():
            chosen.setdefault(nm, {})[str(fold)] = b
        print(f"fold {fold}: loaded from cache", flush=True)
        continue
    print(f"fold {fold}: fit {len(fit)}, val {len(val)}, test {len(test)}  [{time.time() - t0:.0f}s]", flush=True)
    # trivial references
    preds.setdefault("mean", {})[fold] = np.full(len(test), yc[train_all].mean())
    preds.setdefault("last_growth", {})[fold] = d.loc[test, "gdp_growth"].to_numpy(float)
    for kind, gname in settings:
        X = placebo[tuple([gname.split("|placebo")[0], int(gname.split("|placebo")[1])])] if "|placebo" in gname else designs[gname]
        name = f"{kind}|{gname}"
        p, best = fit_predict(kind, X, fit, val, train_all, test)
        preds.setdefault(name, {})[fold] = p
        chosen.setdefault(name, {})[str(fold)] = best
    cache[fold] = {"preds": {nm: v[fold] for nm, v in preds.items()}, "chosen": {nm: v[str(fold)] for nm, v in chosen.items()}}
    pickle.dump(cache, open(CACHE, "wb"))

test_idx = np.concatenate(rows_test)
P = pd.DataFrame({name: np.concatenate([v[f] for f in FOLDS]) for name, v in preds.items()}, index=test_idx)
meta = d.loc[test_idx, ["iso3", "year", "target_year", "target"]].copy()
meta["target_clip"] = np.concatenate(clip_vals)
meta["fold"] = np.concatenate([[f[0]] * len(rows_test[i]) for i, f in enumerate(FOLDS)])
pd.concat([meta, P], axis=1).to_csv(G.OUT / f"preds_baselines_{SAMPLE}.csv", index=False)

# ── metrics ──────────────────────────────────────────────────────────
yt_raw, yt, yr, ct = meta["target"].to_numpy(), meta["target_clip"].to_numpy(), meta["target_year"].to_numpy(), meta["iso3"].to_numpy()
mask_no2020 = yr != 2020


def metrics(name):
    p = P[name].to_numpy()
    out = {}
    for tag, m in (("all", np.ones(len(p), bool)), ("no2020", mask_no2020)):
        out[f"rmse_dm_{tag}"] = float(np.sqrt(np.mean(G.squared_errors(p[m], yt[m], yr[m]))))
        out[f"pinball_dm_{tag}"] = float(np.mean(G.pinball(p[m], yt[m], yr[m])))
        out[f"rmse_dm_{tag}_unclipped"] = float(np.sqrt(np.mean(G.squared_errors(p[m], yt_raw[m], yr[m]))))
        out[f"pinball_dm_{tag}_unclipped"] = float(np.mean(G.pinball(p[m], yt_raw[m], yr[m])))
    return out


M = {name: metrics(name) for name in P.columns}
print(f"\n{'model':<34}{'RMSE dm':>9}{'pinball dm':>12}{'(no 2020: RMSE dm)':>20}")
for name in P.columns:
    if "placebo" in name:
        continue
    m = M[name]
    print(f"{name:<34}{m['rmse_dm_all']:>9.3f}{m['pinball_dm_all']:>12.3f}{m['rmse_dm_no2020']:>20.3f}")

# ── graph vs no-graph comparisons with country-block bootstrap ───────────
def loss_vec(name, quantile=False):
    p = P[name].to_numpy()
    return G.pinball(p, yt, yr) if quantile else G.squared_errors(p, yt, yr)


comparisons = {}
for kind in ("ridge", "rf", "gbm", "gbmq"):
    q = kind in QUANTILE_KINDS
    key = "pinball_dm_all" if q else "rmse_dm_all"
    for gname in GRAPHS:
        a, b = f"{kind}|{gname}", f"{kind}|none"
        diff, lo, hi = G.block_bootstrap_diff(loss_vec(a, q), loss_vec(b, q), ct, root=not q)
        pl = [M[f"{kind}|{gname}|placebo{k}"][key] for k in range(N_PLACEBO)]
        real_gain = M[b][key] - M[a][key]
        pl_gain = M[b][key] - float(np.mean(pl))
        comparisons[f"{kind}|{gname}"] = {
            "metric": key, "no_graph": M[b][key], "with_graph": M[a][key], "diff_graph_minus_none": diff,
            "boot90_lo": lo, "boot90_hi": hi, "placebo_mean": float(np.mean(pl)),
            "real_gain": float(real_gain), "placebo_gain": float(pl_gain),
            "share_of_gain_kept_by_placebo": (float(pl_gain / real_gain) if real_gain > 1e-12 else None),
        }
print("\nGraph features vs no graph (negative = graph helps), 90% country-block bootstrap:")
for k, v in comparisons.items():
    print(f"  {k:<18} {v['metric']:<14} diff {v['diff_graph_minus_none']:+.4f} [{v['boot90_lo']:+.4f}, {v['boot90_hi']:+.4f}] | "
          f"placebo mean {v['placebo_mean']:.4f} vs real {v['with_graph']:.4f}")

json.dump({"sample": SAMPLE, "n_economies": len(ISO), "n_rows_test": int(len(P)), "folds": [list(f) for f in FOLDS],
           "n_placebo_draws": N_PLACEBO, "winsor_bounds_by_fold": bounds, "metrics": M, "comparisons": comparisons, "chosen_hyperparameters": chosen,
           "runtime_seconds": time.time() - t0},
          open(G.OUT / f"baselines_{SAMPLE}.json", "w"), indent=2, default=float)
print(f"\nsaved outputs/baselines_{SAMPLE}.json  ({time.time() - t0:.0f}s)")
