"""
12_spatial_diagnostics.py
════════════════════════════════════════════════════════════════════════
Layer 12 — Cross-Country Dependence and Spatial Spillovers

MOTIVATION
──────────
Every earlier module treats the 18 countries as independent units once
country and year fixed effects are removed. That assumption matters in two
places: the clustered / bootstrap inference of Modules 02-05 (clusters must
be independent) and the Synthetic Control of Module 09 (donors must not be
affected by the treatment). It is also substantively questionable for this
topic: violence, drug-trafficking routes, migration and institutional
shocks cross borders. A graph-neural-network view of the problem would let
information flow between neighbouring countries; the econometric analogue
that is estimable with 18 countries is a spatial weights matrix W.

This module asks, in order:

  1. Is there cross-sectional dependence left in the residuals of the
     paper's equations? (Pesaran CD test; Moran's I per year on residuals,
     permutation inference, since N = 18 is far too small for asymptotics)
  2. Does it follow geography? (residual correlation of neighbouring vs
     non-neighbouring country pairs, permutation test)
  3. Do neighbours' variables affect a country's own outcome? (SLX / spatial
     lag-of-X models: add W·x to each equation; wild cluster bootstrap)
  4. Is the El Salvador synthetic control contaminated by neighbouring
     donors? (weight the synthetic control puts on land neighbours)

SPATIAL WEIGHTS
───────────────
  contiguity  : shared land border (Bahamas is an island -> no neighbours,
                dropped from contiguity-based models)
  inv. dist.  : 1 / great-circle distance between capitals, all pairs
  kNN-4       : the four nearest capitals
The neighbour-pair test needs a yes/no notion of adjacency, so it uses three symmetric
relations instead: land border, either country among the other's four nearest capitals,
and capitals within 1,500 km (the distance-band counterpart of the dense inverse-distance W).
All row-standardised. Trade-based weights would be the natural fourth
choice (bilateral trade is what the GNN trade literature uses) but need an
external bilateral data set (e.g. CEPII BACI/Gravity) that is not part of
this repository.

WHAT THIS MODULE DOES NOT DO
────────────────────────────
It does not estimate spatial-autoregressive (SAR) models: with an
unbalanced panel and N = 18 the Lee-Yu fixed-effects ML estimator is not
reliable, and the diagnostics below decide whether it would be needed.

References
──────────────────────────────────────────────────────────────────────
Pesaran, M. H. (2004). General diagnostic tests for cross section
  dependence in panels. Cambridge Working Papers in Economics 0435.
Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
  Biometrika, 37(1/2), 17-23.
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.
LeSage, J., & Pace, R. K. (2009). Introduction to Spatial Econometrics.
  CRC Press.
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, bold, save_json,
    make_output_dirs, build_multiindex, fe_wild_test, ENTITY_COL, TIME_COL,
)

set_plot_style()
HERE = Path(__file__).parent
DIRS = make_output_dirs(HERE)
BLUE, RED_C, GREY, AMBER = "#2563EB", "#DC2626", "#9CA3AF", "#D97706"
B_BOOT = 999
N_PERM = 4999
MIN_OVERLAP = 8          # minimum common years for a pairwise correlation

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("12-A — LOAD")
df = pd.read_csv(HERE / "panel_enriched.csv")
with open(HERE / "json" / "01_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
CONTROLS = [c for c in meta["controls"] if c != "year_c"]
COUNTRIES = sorted(df[ENTITY_COL].unique())
N = len(COUNTRIES)
IDX = {c: i for i, c in enumerate(COUNTRIES)}
print(ok(f"Panel: {len(df)} obs | {N} countries"))

# ════════════════════════════════════════════════════════════════════════
# SPATIAL WEIGHTS
# ════════════════════════════════════════════════════════════════════════
section("12-B — SPATIAL WEIGHTS MATRICES")

# Approximate capital-city coordinates (lat, lon), used only for distances.
CAPITALS = {
    "BHS": (25.05, -77.35), "BLZ": (17.25, -88.77), "BRA": (-15.79, -47.88),
    "CHL": (-33.45, -70.67), "COL": (4.71, -74.07), "CRI": (9.93, -84.08),
    "DOM": (18.49, -69.93), "ECU": (-0.18, -78.47), "GTM": (14.63, -90.51),
    "HND": (14.07, -87.19), "HTI": (18.54, -72.34), "MEX": (19.43, -99.13),
    "NIC": (12.13, -86.25), "PAN": (8.98, -79.52), "PER": (-12.05, -77.04),
    "PRY": (-25.26, -57.58), "SLV": (13.69, -89.19), "URY": (-34.90, -56.16),
}
# Land borders between countries of the panel (undirected).
BORDERS = [
    ("MEX", "GTM"), ("MEX", "BLZ"), ("GTM", "BLZ"), ("GTM", "HND"), ("GTM", "SLV"),
    ("HND", "SLV"), ("HND", "NIC"), ("NIC", "CRI"), ("CRI", "PAN"), ("PAN", "COL"),
    ("COL", "ECU"), ("COL", "PER"), ("COL", "BRA"), ("ECU", "PER"), ("PER", "BRA"),
    ("PER", "CHL"), ("BRA", "PRY"), ("BRA", "URY"), ("HTI", "DOM"),
]
assert set(CAPITALS) == set(COUNTRIES), "capital table does not match the panel countries"


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


DIST = np.zeros((N, N))
for a in COUNTRIES:
    for b in COUNTRIES:
        if a != b:
            DIST[IDX[a], IDX[b]] = haversine(CAPITALS[a], CAPITALS[b])


def row_std(M):
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


A_CONTIG = np.zeros((N, N))
for a, b in BORDERS:
    A_CONTIG[IDX[a], IDX[b]] = A_CONTIG[IDX[b], IDX[a]] = 1
A_INV = np.divide(1.0, DIST, out=np.zeros_like(DIST), where=DIST > 0)
A_KNN = np.zeros((N, N))
for i in range(N):
    near = np.argsort(np.where(np.arange(N) == i, np.inf, DIST[i]))[:4]
    A_KNN[i, near] = 1

WMATS = {"contiguity": row_std(A_CONTIG), "inverse_distance": row_std(A_INV), "knn4": row_std(A_KNN)}
# Pair definitions for the neighbour-pair test (undirected). A pair is "adjacent" if it shares a border,
# if either country is among the other's four nearest capitals (symmetrised kNN), or if the capitals are
# within BAND_KM of each other (the distance-band counterpart of the dense inverse-distance weights).
BAND_KM = 1500.0
ADJ = {
    "contiguity": A_CONTIG,
    "knn4": ((A_KNN + A_KNN.T) > 0).astype(float),
    "within_1500km": ((DIST > 0) & (DIST <= BAND_KM)).astype(float),
}
for _k, _A in ADJ.items():
    assert np.allclose(_A, _A.T), f"adjacency {_k} must be symmetric"
iso = [COUNTRIES[i] for i in range(N) if A_CONTIG[i].sum() == 0]
print(ok(f"{int(A_CONTIG.sum() / 2)} land borders | isolated in contiguity graph: {iso}"))


def spatial_lag(wide: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
    """W·x per year, renormalising over neighbours with data (NaN when none)."""
    x = wide.reindex(COUNTRIES).to_numpy(float)
    m = (~np.isnan(x)).astype(float)
    num = W @ np.nan_to_num(x)
    den = W @ m
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
    return pd.DataFrame(out, index=COUNTRIES, columns=wide.columns)


def add_lag(d: pd.DataFrame, var: str, W: np.ndarray, name: str) -> pd.DataFrame:
    wide = d.reset_index().pivot(index=ENTITY_COL, columns=TIME_COL, values=var)
    lagged = spatial_lag(wide, W).stack().rename(name)
    lagged.index.names = [ENTITY_COL, TIME_COL]
    return d.join(lagged)


# ════════════════════════════════════════════════════════════════════════
# MODELS (identical to Module 02)
# ════════════════════════════════════════════════════════════════════════
d0 = build_multiindex(df)
EQS = {
    "EQ1": ("inst_avg", ["homicide_rate_log_lag1"] + CONTROLS),
    "EQ2": ("fdi_percent_gdp", ["inst_avg"] + CONTROLS),
    "EQ3": ("gdp_growth", ["fdi_percent_gdp", "inst_avg"] + CONTROLS),
}


def residuals(dep, exog, two_way=True):
    w = d0[[dep] + exog].dropna()
    r = PanelOLS(w[dep], w[exog], entity_effects=True, time_effects=two_way).fit(cov_type="clustered", cluster_entity=True)
    return r.idiosyncratic.iloc[:, 0].rename("e")


# ════════════════════════════════════════════════════════════════════════
# 1. CROSS-SECTIONAL DEPENDENCE: PESARAN CD
# ════════════════════════════════════════════════════════════════════════
section("12-C — PESARAN CD TEST ON RESIDUALS")


def resid_wide(e: pd.Series) -> pd.DataFrame:
    return e.reset_index().pivot(index=ENTITY_COL, columns=TIME_COL, values="e").reindex(COUNTRIES)


def pair_corr(wide: pd.DataFrame):
    """Pairwise correlations over common years (>= MIN_OVERLAP) and overlaps."""
    x = wide.to_numpy(float)
    rho = np.full((N, N), np.nan)
    tij = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            m = ~np.isnan(x[i]) & ~np.isnan(x[j])
            if m.sum() >= MIN_OVERLAP:
                rho[i, j] = rho[j, i] = np.corrcoef(x[i, m], x[j, m])[0, 1]
                tij[i, j] = tij[j, i] = m.sum()
    return rho, tij


def pesaran_cd(rho, tij):
    iu = np.triu_indices(N, 1)
    r, t = rho[iu], tij[iu]
    ok_ = ~np.isnan(r)
    n_pairs = int(ok_.sum())
    cd = np.sqrt(2.0 / (N * (N - 1))) * np.sum(np.sqrt(t[ok_]) * r[ok_])
    # Removing year effects from N units forces residual correlations to average
    # -1/(N-1) (here -0.059): the unadjusted two-way CD is biased negative. The
    # adjusted statistic re-centres each pairwise correlation by that amount.
    cd_adj = np.sqrt(2.0 / (N * (N - 1))) * np.sum(np.sqrt(t[ok_]) * (r[ok_] + 1.0 / (N - 1)))
    return (float(cd), float(2 * stats.norm.sf(abs(cd))), float(np.mean(r[ok_])), float(np.mean(np.abs(r[ok_]))), n_pairs,
            float(cd_adj), float(2 * stats.norm.sf(abs(cd_adj))))


cd_results, wides, rhos = {}, {}, {}
for name, (dep, exog) in EQS.items():
    for tag, two_way in (("two_way", True), ("entity_only", False)):
        e = residuals(dep, exog, two_way)
        wide = resid_wide(e)
        rho, tij = pair_corr(wide)
        cd, p, mean_r, mean_abs, npairs, cd_adj, p_adj = pesaran_cd(rho, tij)
        cd_results[f"{name}_{tag}"] = {"cd": cd, "p": p, "cd_adjusted": cd_adj, "p_adjusted": p_adj,
                                       "mean_rho": mean_r, "mean_abs_rho": mean_abs, "n_pairs": npairs}
        if two_way:
            wides[name], rhos[name] = wide, rho
        extra = f" | adjusted for -1/(N-1): CD = {cd_adj:+6.2f} (p = {p_adj:.3f})" if two_way else ""
        print(f"    {name} {tag:<11}: CD = {cd:+6.2f} (p = {p:.3f}) | mean rho = {mean_r:+.3f}{extra}")
print(ok("Entity-only residuals still contain the common year shocks; two-way residuals have them removed (at the price of the -1/(N-1) artefact)."))

# ════════════════════════════════════════════════════════════════════════
# 2. DOES DEPENDENCE FOLLOW GEOGRAPHY?
# ════════════════════════════════════════════════════════════════════════
section("12-D — MORAN'S I AND NEIGHBOUR-PAIR CORRELATION (PERMUTATION)")
rng = np.random.default_rng(42)


PERMS = np.stack([rng.permutation(N) for _ in range(N_PERM)])          # shared label permutations
IU = np.triu_indices(N, 1)


def permute(M, perms):
    """Stack of M with rows and columns relabelled by each permutation: (P, N, N)."""
    return M[perms[:, :, None], perms[:, None, :]]


def morans_i_avg(wide, Wstack):
    """Average over years of Moran's I of the residual cross-section, for a stack of weights matrices (P, N, N).

    Each year uses only the countries with a residual, and the weights are re-standardised on them."""
    x = wide.to_numpy(float)
    per_year = []
    for t in range(x.shape[1]):
        idx = np.where(~np.isnan(x[:, t]))[0]
        if len(idx) < 8:
            continue
        z = x[idx, t] - x[idx, t].mean()
        den = float(z @ z)
        if den <= 0:
            continue
        sub = Wstack[:, idx][:, :, idx]
        rs = sub.sum(axis=2, keepdims=True)
        sub = np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)
        s0 = sub.sum(axis=(1, 2))
        num = np.einsum("i,pij,j->p", z, sub, z)
        per_year.append(np.where(s0 > 0, len(idx) / np.where(s0 > 0, s0, 1) * num / den, np.nan))
    return np.nanmean(np.vstack(per_year), axis=0) if per_year else np.full(Wstack.shape[0], np.nan)


def moran_perm(wide, W):
    obs = float(morans_i_avg(wide, W[None])[0])
    sims = morans_i_avg(wide, permute(W, PERMS))
    return obs, float((np.sum(sims >= obs) + 1) / (len(sims) + 1)), float(np.mean(sims))


def neighbour_gap(rho, A):
    """Mean residual correlation of adjacent pairs minus non-adjacent pairs (A must be symmetric)."""
    r = rho[IU]
    keep = (~np.isnan(r)).astype(float)
    r0 = np.nan_to_num(r)

    def stat(adj):                                  # adj: (..., npairs) in {0, 1}
        a, b = adj * keep, (1 - adj) * keep
        return (a * r0).sum(-1) / a.sum(-1) - (b * r0).sum(-1) / b.sum(-1)

    adj_obs = A[IU]
    gap_obs = float(stat(adj_obs))
    sims = stat(permute(A, PERMS)[:, IU[0], IU[1]])
    return gap_obs, float((np.sum(sims >= gap_obs) + 1) / (len(sims) + 1)), int((adj_obs * keep).sum())


moran, gap = {}, {}
for name in EQS:
    for wn, W in WMATS.items():
        i_obs, p_i, i_null = moran_perm(wides[name], W)
        moran[f"{name}_{wn}"] = {"I": i_obs, "p_perm": p_i, "I_null_mean": i_null}
        print(f"    {name} | Moran's I with {wn:<16}: I = {i_obs:+.3f} (permutation p = {p_i:.3f})")
    for an, A in ADJ.items():
        g, p_g, n_adj = neighbour_gap(rhos[name], A)
        gap[f"{name}_{an}"] = {"gap": g, "p_perm": p_g, "n_adjacent_pairs": n_adj}
        print(f"    {name} | neighbour pairs ({an:<13}): mean residual corr. minus other pairs = {g:+.3f} (p = {p_g:.3f}, {n_adj} adjacent pairs)")

# ════════════════════════════════════════════════════════════════════════
# 3. SPILLOVER (SLX) MODELS
# ════════════════════════════════════════════════════════════════════════
section("12-E — SPILLOVER MODELS (SLX: own variables + neighbours' variables)")
SLX_SPECS = [
    # label, dependent, own key regressor, neighbour variable (its W-lag is added)
    ("EQ0 violence contagion", "homicide_rate_log", "homicide_rate_log_lag1", "homicide_rate_log_lag1", [], True),
    ("EQ1 neighbours' violence → own institutions", "inst_avg", "homicide_rate_log_lag1", "homicide_rate_log_lag1", CONTROLS, False),
    ("EQ2 neighbours' institutions → own FDI", "fdi_percent_gdp", "inst_avg", "inst_avg", CONTROLS, False),
    ("EQ3 neighbours' FDI → own growth", "gdp_growth", "fdi_percent_gdp", "fdi_percent_gdp", CONTROLS + ["inst_avg"], False),
    ("EQ3b neighbours' growth → own growth", "gdp_growth", "gdp_growth_lag1", "gdp_growth_lag1", CONTROLS, False),
]
d0["gdp_growth_lag1"] = d0.groupby(level=0)["gdp_growth"].shift(1)

slx = {}
for label, dep, own, nb, ctrl, _ in SLX_SPECS:
    for wn in ("inverse_distance", "contiguity"):
        W = WMATS[wn]
        wname = f"W_{nb}"
        d = add_lag(d0, nb, W, wname)
        exog = list(dict.fromkeys([own] + ctrl + [wname]))
        w = d[[dep] + exog].dropna()
        r = fe_wild_test(w, dep, wname, exog, B=B_BOOT)
        r_own = fe_wild_test(w, dep, own, exog, B=B_BOOT)
        slx[f"{label}|{wn}"] = {"label": label, "weights": wn, "dep": dep, "neighbour_var": nb,
                                 "theta": r["coef"], "p_cl": r["p_cl"], "p_wild": r["p_wild"],
                                 "own_coef": r_own["coef"], "own_p_wild": r_own["p_wild"], "n_obs": r["n_obs"]}
        print(f"    {label:<46} {wn:<16}: theta = {r['coef']:+.3f} (p_cl {r['p_cl']:.3f}, p_wild {r['p_wild']:.3f}) | own = {r_own['coef']:+.3f} (p_wild {r_own['p_wild']:.3f}) | n = {r['n_obs']}")

n_tests = len(slx)
n_sig = sum(v["p_wild"] < 0.05 for v in slx.values())
print(ok(f"{n_sig} of {n_tests} spillover coefficients significant at 5% (wild bootstrap); about {0.05 * n_tests:.1f} expected by chance"))

# ════════════════════════════════════════════════════════════════════════
# 4. SYNTHETIC CONTROL CONTAMINATION CHECK
# ════════════════════════════════════════════════════════════════════════
section("12-F — IS THE SYNTHETIC EL SALVADOR CONTAMINATED BY NEIGHBOURS?")
with open(HERE / "json" / "09_synthetic_control.json", encoding="utf-8") as f:
    sc = json.load(f)
w_sc = sc["el_salvador"]["weights"]
nb_contig = [COUNTRIES[j] for j in range(N) if A_CONTIG[IDX["SLV"], j] > 0]
near_1500 = [COUNTRIES[j] for j in range(N) if j != IDX["SLV"] and DIST[IDX["SLV"], j] <= 1500]
sc_check = {
    "land_neighbours": nb_contig,
    "weight_on_land_neighbours": float(sum(w_sc.get(c, 0.0) for c in nb_contig)),
    "within_1500km": near_1500,
    "weight_within_1500km": float(sum(w_sc.get(c, 0.0) for c in near_1500)),
    "positive_weights": {k: v for k, v in w_sc.items() if v > 1e-6},
}
print(f"    land neighbours of El Salvador in the panel: {nb_contig}")
print(f"    synthetic-control weight on them: {sc_check['weight_on_land_neighbours']:.3f}")
print(f"    weight on donors within 1,500 km {near_1500}: {sc_check['weight_within_1500km']:.3f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("12-G — FIGURE")
order = ["MEX", "BLZ", "GTM", "SLV", "HND", "NIC", "CRI", "PAN", "COL", "ECU", "PER", "BRA", "CHL", "PRY", "URY", "BHS", "HTI", "DOM"]
pos = [IDX[c] for c in order]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
R = rhos["EQ1"][np.ix_(pos, pos)]
im = ax[0].imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(N)); ax[0].set_xticklabels(order, rotation=90, fontsize=8)
ax[0].set_yticks(range(N)); ax[0].set_yticklabels(order, fontsize=8)
for a, b in BORDERS:
    i, j = order.index(a), order.index(b)
    for (u, v) in ((i, j), (j, i)):
        ax[0].add_patch(plt.Rectangle((v - .5, u - .5), 1, 1, fill=False, ec="black", lw=1.2))
ax[0].set_title("EQ1 residual correlations (black frame = land border)", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=0.046)
labels = [f"{eq}\n{an.replace('_', ' ')}" for eq in EQS for an in ADJ]
vals = [gap[f"{eq}_{an}"]["gap"] for eq in EQS for an in ADJ]
ps = [gap[f"{eq}_{an}"]["p_perm"] for eq in EQS for an in ADJ]
ax[1].barh(range(len(vals)), vals, color=[BLUE if p < 0.05 else GREY for p in ps])
ax[1].set_yticks(range(len(vals))); ax[1].set_yticklabels(labels, fontsize=7)
ax[1].axvline(0, color="black", lw=0.8)
ax[1].set_xlabel("mean residual correlation: neighbouring pairs minus other pairs")
for i, (v, p) in enumerate(zip(vals, ps)):
    ax[1].text(v, i, f"  p={p:.2f}", va="center", fontsize=7)
ax[1].set_title("Does residual dependence follow geography? (blue: perm. p<0.05)", fontsize=10)
ax[1].invert_yaxis()
fig.tight_layout()
fig.savefig(DIRS["figures"] / "17_spatial_dependence.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/17_spatial_dependence.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
n_cd_sig = sum(v["p_adjusted"] < 0.05 for k, v in cd_results.items() if k.endswith("two_way"))
n_moran_sig = sum(v["p_perm"] < 0.05 for v in moran.values())
n_gap_sig = sum(v["p_perm"] < 0.05 for v in gap.values())
save_json({
    "n_countries": N,
    "weights": {"land_borders": int(A_CONTIG.sum() / 2), "isolated_contiguity": iso,
                "adjacent_pairs": {k: int(A.sum() / 2) for k, A in ADJ.items()}, "band_km": BAND_KM,
                "capital_coordinates_approximate": True, "trade_weights": "not used (needs external bilateral data)"},
    "pesaran_cd": cd_results,
    "morans_i": moran,
    "neighbour_pair_gap": gap,
    "slx": slx,
    "summary": {
        "n_cd_two_way_significant_05": int(n_cd_sig), "n_cd_two_way": 3,
        "n_moran_significant_05": int(n_moran_sig), "n_moran": len(moran),
        "n_gap_significant_05": int(n_gap_sig), "n_gap": len(gap),
        "n_slx_significant_05": int(n_sig), "n_slx": n_tests,
        "n_permutations": N_PERM, "n_bootstrap": B_BOOT,
    },
    "synthetic_control_contamination": sc_check,
}, DIRS["json"] / "12_spatial_diagnostics.json")
print(bold(f"\nSummary: CD significant (two-way) {n_cd_sig}/3 | Moran {n_moran_sig}/{len(moran)} | neighbour gap {n_gap_sig}/{len(gap)} | SLX {n_sig}/{n_tests}"))
