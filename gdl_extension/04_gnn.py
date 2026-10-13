"""
04_gnn.py
════════════════════════════════════════════════════════════════════════
GDL extension, step 4 -- graph neural networks and their no-graph twin.

Per feature year t the economies form a graph snapshot: node features X_t (same features and imputation as
the baselines), graph fixed over time (trade 2000-2008, inverse distance, 4 nearest, borders), target the
clipped, year-demeaned growth of t+1. Models (small by design, ~3-5k parameters):

  mlp   no graph (same capacity, same training procedure)
  gcn   two GCNConv layers (edge weights = row-normalised graph weights)
  gat   two GATConv layers (4 heads)

Each is trained for the mean (MSE) and for the growth ceiling (pinball loss at 0.90). Training: AdamW,
early stopping on the last three target years of the training window, then re-training from scratch on the
whole training window for the selected number of epochs. 3 seeds for real graphs; placebo graphs are
node permutations (one run per draw). Results are cached after every fit, so an interrupted run resumes.

Run in the GDL environment:  .venv-gdl\\Scripts\\python 04_gnn.py [primary|nohom] [--quick]
Predictions go to outputs/preds_gnn_<sample>.csv (seed-averaged); per-seed metrics to outputs/gnn_<sample>.json.
════════════════════════════════════════════════════════════════════════
"""
import json
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

import gdl_common as G

warnings.filterwarnings("ignore")
torch.set_num_threads(2)                       # leave CPU for the rest of the machine
SAMPLE = next((a for a in sys.argv[1:] if a in ("primary", "nohom")), "primary")
QUICK = "--quick" in sys.argv
N_SEEDS = 1 if QUICK else 3
N_PLACEBO = 1 if QUICK else 5
FOLDS = G.FOLDS[-1:] if QUICK else G.FOLDS
HIDDEN, DROPOUT, LR, WD, MAX_EPOCHS, PATIENCE = 32, 0.2, 5e-3, 5e-4, 200, 20
TARGETS = ("mean", "ceiling")

d, FEATS = G.load_dataset(SAMPLE, require_target=False)
ISO = sorted(d["iso3"].unique())
N = len(ISO)
IDX = {c: i for i, c in enumerate(ISO)}
GRAPHS = G.build_graphs(ISO)
YEARS = np.arange(2000, 2024)
T = len(YEARS)
node = d["iso3"].map(IDX).to_numpy()
tix = (d["year"] - 2000).to_numpy()
y_raw = d["target"].to_numpy(float)
years_t = d["target_year"].to_numpy()
print(f"sample={SAMPLE}: {N} economies, {len(d)} rows (with target: {int(np.isfinite(y_raw).sum())}), graphs={list(GRAPHS)}, "
      f"seeds={N_SEEDS}, placebo draws={N_PLACEBO}", flush=True)


# ── models ───────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, x, ei, ew):
        return self.net(x).squeeze(-1)


class GCN(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.c1, self.c2, self.out = GCNConv(fin, HIDDEN), GCNConv(HIDDEN, HIDDEN), nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ei, ew):
        h = self.drop(torch.relu(self.c1(x, ei, ew)))
        h = self.drop(torch.relu(self.c2(h, ei, ew)))
        return self.out(h).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.c1, self.c2 = GATConv(fin, HIDDEN // 4, heads=4, dropout=DROPOUT), GATConv(HIDDEN, HIDDEN // 4, heads=4, dropout=DROPOUT)
        self.out = nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ei, ew):
        h = self.drop(torch.nn.functional.elu(self.c1(x, ei)))
        h = self.drop(torch.nn.functional.elu(self.c2(h, ei)))
        return self.out(h).squeeze(-1)


MODELS = {"mlp": MLP, "gcn": GCN, "gat": GAT}


def pinball_t(pred, y):
    e = y - pred
    return torch.maximum(G.QUANTILE * e, (G.QUANTILE - 1) * e).mean()


def edges_of(W):
    ii, jj = np.nonzero(W)
    return torch.tensor(np.vstack([jj, ii]), dtype=torch.long), torch.tensor(W[ii, jj], dtype=torch.float32)


def tile(ei, ew, k):
    """k disjoint copies of the graph (one per year snapshot)."""
    off = (torch.arange(k) * N).repeat_interleave(ei.shape[1])
    return ei.repeat(1, k) + off.unsqueeze(0), ew.repeat(k)


# ── one fold: tensors ────────────────────────────────────────────────
def fold_tensors(fold):
    fit, val, train_all, test = G.fold_indices(d, fold)
    lo, hi = G.winsor_bounds(y_raw[train_all][np.isfinite(y_raw[train_all])])
    yc_raw = np.clip(y_raw, lo, hi)
    ydm = G.demean_by_year(yc_raw, years_t)          # training/validation scale
    Xdf = d[FEATS]
    return fit, val, train_all, test, ydm, yc_raw


def build_inputs(stats_rows, fit_rows_for_impute):
    """Imputed, standardised features for every (year, node) cell. Cells without a row are all-missing."""
    Xr = d[FEATS]
    Xt, (Xall,) = G.impute(Xr.loc[fit_rows_for_impute], [Xr])
    full = pd.DataFrame(np.nan, index=range(T * N), columns=Xall.columns)
    flat = tix * N + node
    full.loc[flat, :] = Xall.to_numpy()
    # cells with no row: training medians, missing indicators = 1
    absent = np.setdiff1d(np.arange(T * N), flat)
    med = Xt.median()
    for c in full.columns:
        full.loc[absent, c] = 1.0 if c.endswith("_missing") else med[c]
    mu, sd = Xt.mean(), Xt.std().replace(0, 1.0)
    return ((full - mu) / sd).to_numpy(np.float32)


def run_fit(model_name, W, target, seed, fold_pack, train_rows, val_rows, eval_rows, X, epochs=None):
    """Train on `train_rows`; with val_rows early-stop and return the best epoch; else train `epochs` and predict eval_rows."""
    fit, val, train_all, test, ydm, _ = fold_pack
    torch.manual_seed(seed)
    np.random.seed(seed)
    fin = X.shape[1]
    model = MODELS[model_name](fin)
    ei, ew = (edges_of(W) if W is not None else (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = pinball_t if target == "ceiling" else (lambda p, yv: ((p - yv) ** 2).mean())

    def snapshot(rows):
        ts = np.unique(tix[rows])
        pos = {t: k for k, t in enumerate(ts)}
        xs = torch.tensor(np.concatenate([X[t * N:(t + 1) * N] for t in ts]))
        sel = np.array([pos[t] * N + n_ for t, n_ in zip(tix[rows], node[rows])])
        ok = np.isfinite(ydm[rows])
        return xs, tile(ei, ew, len(ts)), torch.tensor(sel[ok]), torch.tensor(ydm[rows][ok], dtype=torch.float32), (rows, ok, sel)

    xtr, (ei_tr, ew_tr), sel_tr, ytr, _ = snapshot(train_rows)
    xv = None
    if val_rows is not None:
        xv, (ei_v, ew_v), sel_v, yv, _ = snapshot(val_rows)
    best, best_ep, since = np.inf, 0, 0
    n_ep = MAX_EPOCHS if val_rows is not None else epochs
    for ep in range(1, n_ep + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(xtr, ei_tr, ew_tr)[sel_tr], ytr)
        loss.backward()
        opt.step()
        if val_rows is not None:
            model.eval()
            with torch.no_grad():
                vl = float(lossf(model(xv, ei_v, ew_v)[sel_v], yv))
            if vl < best - 1e-6:
                best, best_ep, since = vl, ep, 0
            else:
                since += 1
                if since >= PATIENCE:
                    break
    if val_rows is not None:
        return best_ep, best
    xe, (ei_e, ew_e), _, _, (rows, ok, sel) = snapshot(eval_rows)
    model.eval()
    with torch.no_grad():
        out = model(xe, ei_e, ew_e).numpy()
    return out[sel]


# ── main loop with cache ─────────────────────────────────────────────
CACHE = G.OUT / f"gnn_cache_{SAMPLE}{'_quick' if QUICK else ''}.pkl"
cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}


def settings():
    s = [("mlp", "none", None)]
    for g in GRAPHS:
        s += [("gcn", g, None), ("gat", g, None)]
    for g in GRAPHS:
        for k in range(N_PLACEBO):
            s += [("gcn", g, k), ("gat", g, k)]
    return s


t0 = time.time()
n_done = 0
for fold in FOLDS:
    pack = fold_tensors(fold)
    fit, val, train_all, test, ydm, _ = pack
    X_val_stage = build_inputs(None, fit)            # imputer / scaler fitted on the fit rows
    X_final = build_inputs(None, train_all)          # fitted on the whole training window
    print(f"fold {fold}: fit {len(fit)}, val {len(val)}, train {len(train_all)}, test {len(test)}  [{time.time() - t0:.0f}s]", flush=True)
    for (mname, gname, plc) in settings():
        W = None
        if gname != "none":
            W = GRAPHS[gname]
            if plc is not None:
                W = G.permute_graph(W, np.random.default_rng(7000 + 100 * list(GRAPHS).index(gname) + plc))
        seeds = range(N_SEEDS) if plc is None else [plc]
        for target in TARGETS:
            ekey = ("E", mname, gname, target, str(fold))
            if ekey not in cache:                    # epochs selected once per setting, on the REAL graph with seed 0
                W_real = None if gname == "none" else GRAPHS[gname]
                E0, v0 = run_fit(mname, W_real, target, 0, pack, fit, val, None, X_val_stage)
                cache[ekey] = (int(E0), float(v0))
            E = cache[ekey][0]
            for seed in seeds:
                key = (mname, gname, plc, target, seed, str(fold))
                if key in cache:
                    continue
                pred = run_fit(mname, W, target, seed, pack, train_all, None, test, X_final, epochs=max(E, 5))
                cache[key] = {"pred": pred, "epochs": int(E)}
                n_done += 1
        pickle.dump(cache, open(CACHE, "wb"))
        if n_done and n_done % 50 == 0:
            print(f"   {n_done} fits [{time.time() - t0:.0f}s]", flush=True)

# ── assemble predictions (seed-averaged) and metrics ────────────────
FITS = [k for k in cache if k[0] != "E"]
test_idx = np.concatenate([G.fold_indices(d, f)[3] for f in FOLDS])
meta = d.loc[test_idx, ["iso3", "year", "target_year", "target"]].reset_index(drop=True)
lo_hi = {str(f): G.winsor_bounds(y_raw[G.fold_indices(d, f)[2]][np.isfinite(y_raw[G.fold_indices(d, f)[2]])]) for f in FOLDS}
meta["target_clip"] = np.concatenate([np.clip(y_raw[G.fold_indices(d, f)[3]], *lo_hi[str(f)]) for f in FOLDS])
meta["fold"] = np.concatenate([[f[0]] * len(G.fold_indices(d, f)[3]) for f in FOLDS])
cols, per_seed = {}, {}
for (mname, gname, plc, target) in sorted({(k[0], k[1], k[2], k[3]) for k in FITS}, key=str):
    name = f"{mname}|{gname}|{target}" + ("" if plc is None else f"|placebo{plc}")
    seeds = sorted({k[4] for k in FITS if k[:4] == (mname, gname, plc, target)})
    mats = []
    for s in seeds:
        mats.append(np.concatenate([cache[(mname, gname, plc, target, s, str(f))]["pred"] for f in FOLDS]))
    cols[name] = np.mean(mats, axis=0)
    per_seed[name] = mats
P = pd.DataFrame(cols)
pd.concat([meta, P], axis=1).to_csv(G.OUT / f"preds_gnn_{SAMPLE}{'_quick' if QUICK else ''}.csv", index=False)

yt, yr = meta["target_clip"].to_numpy(), meta["target_year"].to_numpy()
rows = {}
for name, mats in per_seed.items():
    q = name.split("|")[2] == "ceiling"
    f = (lambda p: float(np.mean(G.pinball(p, yt, yr)))) if q else (lambda p: float(np.sqrt(np.mean(G.squared_errors(p, yt, yr)))))
    vals = [f(m) for m in mats]
    rows[name] = {"metric": "pinball_dm" if q else "rmse_dm", "ensemble": f(P[name].to_numpy()),
                  "per_seed_mean": float(np.mean(vals)), "per_seed_sd": float(np.std(vals)), "n_seeds": len(vals)}
json.dump({"sample": SAMPLE, "n_economies": N, "folds": [list(f) for f in FOLDS], "hyper": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
          "weight_decay": WD, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}, "results": rows,
           "epochs": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}|{k[5]}": v["epochs"] for k, v in cache.items() if k[0] != "E"}},
          open(G.OUT / f"gnn_{SAMPLE}{'_quick' if QUICK else ''}.json", "w"), indent=2)
print(f"\n{'setting':<30}{'ensemble':>10}{'per-seed mean':>15}{'sd':>8}")
for name, r in rows.items():
    if "placebo" not in name:
        print(f"{name:<30}{r['ensemble']:>10.3f}{r['per_seed_mean']:>15.3f}{r['per_seed_sd']:>8.3f}")
print(f"\nsaved outputs/gnn_{SAMPLE}.json  ({time.time() - t0:.0f}s)")
