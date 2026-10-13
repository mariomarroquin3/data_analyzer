"""
gdl_common.py -- shared data, validation scheme, graphs and metrics for the GDL extension.

Everything that defines the evaluation (samples, features, folds, imputation, metrics, bootstrap)
lives here so that the baselines and the graph neural networks are scored by exactly the same code.
The rules are the ones written in README.md before any model was fitted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

WGI = ["rule_of_law", "political_stability", "control_corruption", "voice_accountability",
       "government_effectiveness", "regulatory_quality"]
CORE = ["gdp_growth", "log_gdp_pc", "fdi_percent_gdp", "inflation_t", "exports_percent_gdp", "log_population"]
FEATURES_PRIMARY = CORE + ["log_homicide"] + WGI
FEATURES_NOHOM = CORE + WGI
# rolling-origin test blocks, as FEATURE years (the target is growth one year later)
FOLDS = [(2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023)]
GAP = 1                      # one feature year left out between training and test
N_VAL_YEARS = 3             # last target years of each training window are used for tuning only
QUANTILE = 0.90


# ───────────────────────────── data ─────────────────────────────
def load_dataset(sample="primary", require_target=True):
    """Rows = (economy, feature year t) with the target gdp_growth(t+1). Returns (df, features).
    With require_target=False the rows whose next-year growth is missing are kept (their features still serve
    as neighbours in graph models); the baselines and the evaluation only use rows with a target."""
    panel = pd.read_csv(DATA / "global_panel.csv")
    cov = pd.read_csv(DATA / "coverage_by_country.csv").set_index("iso3")
    core_ok = cov[[f"n_{v}" for v in ("gdp_growth", "log_gdp_pc", "fdi_percent_gdp", "inflation", "exports_percent_gdp", "log_population")]].min(axis=1) >= 20
    wgi_ok = cov[[f"n_{v}" for v in WGI]].min(axis=1) >= 20
    hom_ok = cov["n_log_homicide"] >= 15
    keep = core_ok & wgi_ok & (hom_ok if sample == "primary" else True)
    iso = sorted(keep[keep].index)
    d = panel[panel["iso3"].isin(iso)].copy()
    # declared transformations of heavy-tailed features (the target is NOT transformed)
    d["inflation_t"] = np.sign(d["inflation"]) * np.log1p(d["inflation"].abs())
    d["fdi_percent_gdp"] = d["fdi_percent_gdp"].clip(-30, 60)
    d["gdp_growth_feat"] = d["gdp_growth"].clip(-30, 30)
    nxt = d[["iso3", "year", "gdp_growth"]].copy()
    nxt["year"] -= 1
    d = d.merge(nxt.rename(columns={"gdp_growth": "target"}), on=["iso3", "year"], how="left")
    d["target_year"] = d["year"] + 1
    d["gdp_growth"] = d["gdp_growth_feat"]
    feats = FEATURES_PRIMARY if sample == "primary" else FEATURES_NOHOM
    d = d[d["year"].between(2000, 2023)]
    if require_target:
        d = d[d["target"].notna()]
    d = d.reset_index(drop=True)
    return d, feats


# ───────────────────────────── graphs ─────────────────────────────
TOPK = 10


def _topk_rows(W, k=TOPK):
    """Keep each row's k largest weights (dense graphs would give every node every other node as a neighbour)."""
    out = np.zeros_like(W)
    for i in range(len(W)):
        j = np.argsort(W[i])[::-1][:k]
        j = j[W[i, j] > 0]
        out[i, j] = W[i, j]
    return out


def _row_norm(W):
    s = W.sum(axis=1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def build_graphs(iso):
    """Row-normalised weights matrices among the sample economies, keyed by graph name."""
    idx = {c: i for i, c in enumerate(iso)}
    n = len(iso)
    edges = pd.read_csv(DATA / "edges_static.csv")
    edges = edges[edges["iso_o"].isin(idx) & edges["iso_d"].isin(idx)]
    r, c = edges["iso_o"].map(idx).to_numpy(), edges["iso_d"].map(idx).to_numpy()
    graphs = {}
    border = np.zeros((n, n)); border[r, c] = edges["border"].to_numpy(float)
    graphs["border"] = _row_norm(np.maximum(border, border.T))
    dist = np.full((n, n), np.nan); dist[r, c] = edges["distw"].to_numpy(float)
    dist = np.where(np.isnan(dist), dist.T, dist)
    inv = np.divide(1.0, dist, out=np.zeros_like(dist), where=np.isfinite(dist) & (dist > 0))
    graphs["inv_distance"] = _row_norm(_topk_rows(inv))      # the 10 nearest, weights proportional to 1/distance
    knn = np.zeros((n, n))
    for i in range(n):
        d_i = np.where(np.isfinite(dist[i]), dist[i], np.inf); d_i[i] = np.inf
        knn[i, np.argsort(d_i)[:4]] = 1.0
    graphs["knn4"] = _row_norm(knn)
    tpath = DATA / "edges_trade.csv"
    if tpath.exists():
        tr = pd.read_csv(tpath)
        tr = tr[tr["iso_o"].isin(idx) & tr["iso_d"].isin(idx)]
        T = np.zeros((n, n)); T[tr["iso_o"].map(idx).to_numpy(), tr["iso_d"].map(idx).to_numpy()] = tr["flow_kusd"].to_numpy(float)
        graphs["trade"] = _row_norm(_topk_rows(T + T.T))          # the 10 largest trade partners (exports + imports)
    return graphs


def neighbour_features(d, feats, W, iso):
    """Weighted neighbour average of each feature in the same year (renormalised over neighbours with data;
    isolated economies and years with no neighbour data fall back to that year's cross-country mean)."""
    idx = {c: i for i, c in enumerate(iso)}
    out = pd.DataFrame(index=d.index, columns=[f"nb_{f}" for f in feats], dtype=float)
    for year, g in d.groupby("year"):
        X = np.full((len(iso), len(feats)), np.nan)
        X[g["iso3"].map(idx).to_numpy()] = g[feats].to_numpy(float)
        m = (~np.isnan(X)).astype(float)
        num, den = W @ np.nan_to_num(X), W @ m
        nb = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
        fallback = np.nanmean(X, axis=0)
        nb = np.where(np.isnan(nb), fallback[None, :], nb)
        out.loc[g.index] = nb[g["iso3"].map(idx).to_numpy()]
    return out


def permute_graph(W, rng):
    """Placebo: relabel the nodes (structure and weights preserved, alignment with the countries destroyed)."""
    p = rng.permutation(len(W))
    return W[np.ix_(p, p)]


# ───────────────────────────── folds and imputation ─────────────────────────────
def fold_indices(d, fold):
    t0, t1 = fold
    test = d.index[d["year"].between(t0, t1)]
    train_all = d.index[d["target_year"] <= t0 - GAP]
    val_start = (t0 - GAP) - N_VAL_YEARS + 1           # first target year of the validation block
    fit = d.index[d["target_year"] < val_start]
    val = d.index[(d["target_year"] >= val_start) & (d["target_year"] <= t0 - GAP)]
    assert set(train_all) == set(fit) | set(val)
    return fit, val, train_all, test


def impute(X_train, X_other):
    """Median fill from the training rows (per column) plus missing indicators for columns incomplete in training."""
    med = X_train.median()
    med = med.fillna(0.0)
    miss_cols = [c for c in X_train.columns if X_train[c].isna().any()]

    def f(X):
        Xf = X.fillna(med)
        for c in miss_cols:
            Xf[c + "_missing"] = X[c].isna().astype(float).to_numpy()
        return Xf
    return f(X_train), [f(x) for x in X_other]


# ───────────────────────────── metrics ─────────────────────────────
def winsor_bounds(y_train, lo_q=0.01, hi_q=0.99):
    """Bounds of the 1st-99th percentile range of the TRAINING window's targets. Training targets, validation
    targets and the evaluated test targets of a fold are all clipped to these bounds (no test information used);
    the unclipped metrics are reported as secondary (see README, deviations log)."""
    lo, hi = np.quantile(np.asarray(y_train, float), [lo_q, hi_q])
    return float(lo), float(hi)


def demean_by_year(values, years):
    s = pd.Series(np.asarray(values, float))
    return (s - s.groupby(np.asarray(years)).transform("mean")).to_numpy()


def squared_errors(pred, actual, years, demeaned=True):
    if demeaned:
        pred, actual = demean_by_year(pred, years), demean_by_year(actual, years)
    return (np.asarray(pred) - np.asarray(actual)) ** 2


def pinball(pred, actual, years, q=QUANTILE, demeaned=True):
    if demeaned:
        pred, actual = demean_by_year(pred, years), demean_by_year(actual, years)
    e = np.asarray(actual) - np.asarray(pred)
    return np.maximum(q * e, (q - 1) * e)


def block_bootstrap_diff(loss_a, loss_b, countries, B=1000, seed=0, root=True):
    """Country-block bootstrap of RMSE(a) - RMSE(b) (or the mean-loss difference when root=False).
    Negative values mean model a is better."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(countries))
    G = len(uniq)
    sa = np.bincount(codes, weights=loss_a, minlength=G)
    sb = np.bincount(codes, weights=loss_b, minlength=G)
    cnt = np.bincount(codes, minlength=G).astype(float)
    draws = rng.integers(0, G, size=(B, G))
    na = cnt[draws].sum(axis=1)
    ma = sa[draws].sum(axis=1) / na
    mb = sb[draws].sum(axis=1) / na
    d = (np.sqrt(ma) - np.sqrt(mb)) if root else (ma - mb)
    obs = (np.sqrt(sa.sum() / cnt.sum()) - np.sqrt(sb.sum() / cnt.sum())) if root else (sa.sum() / cnt.sum() - sb.sum() / cnt.sum())
    return float(obs), float(np.percentile(d, 5)), float(np.percentile(d, 95))
