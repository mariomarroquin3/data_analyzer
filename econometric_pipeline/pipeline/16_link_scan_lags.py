"""
16_link_scan_lags.py
════════════════════════════════════════════════════════════════════════
Layer 16 — Does the link scan change with longer lags? (exploratory extension of Module 13)

Module 13 tested every ordered pair of nine variables with the cause lagged ONE year.
Here the cause enters lagged 1, 2 or 3 years. The family is fixed in advance:
72 links x 3 lags = 216 tests, and ALL multiplicity control (Benjamini-Hochberg q = 0.10, Holm)
is applied across the whole family of 216, not lag by lag.

Same design as Module 13: country and year fixed effects, the outcome's own (real) one-year lag
as a control, wild cluster bootstrap (G = 18). The lag-1 results reproduce Module 13 exactly.
Placebo calibration: each cause is shuffled within country (its distribution is kept, its timing
destroyed) and the whole 216-test scan is repeated; the share of "significant" links and the
number of BH survivors in those scans are the procedure's own false-positive behaviour.

Chains X -(a years)-> M -(b years)-> Y are examined only when both links survive the 216-test control.
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, ok, warn, bold, save_json, make_output_dirs, build_multiindex,
    fe_wild_test, country_cluster_bootstrap, link_design, ENTITY_COL,
)

set_plot_style()
HERE = Path(__file__).parent
DIRS = make_output_dirs(HERE)
B_BOOT, B_PLACEBO, N_PLACEBO, B_CHAIN = 999, 199, 10, 499
Q_FDR, MIN_OBS = 0.10, 120
LAGS = (1, 2, 3)

df = pd.read_csv(HERE / "panel_enriched.csv")
with open(HERE / "json" / "01_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
CONTROLS = [c for c in meta["controls"] if c != "year_c"]
NODES = {
    "violence": "homicide_rate_log", "institutions": "inst_avg", "FDI": "fdi_percent_gdp", "growth": "gdp_growth",
    "inflation": "inflation", "unemployment": "unemployment", "exports": "exports_percent_gdp",
    "tourism": "tourist_arrivals_log", "remittances": "remittances_percent_gdp",
}
d0 = build_multiindex(df)
for col in NODES.values():
    for L in LAGS:
        d0[f"{col}__lag{L}"] = d0.groupby(level=0)[col].shift(L)
PAIRS = list(itertools.permutations(NODES, 2))
print(ok(f"{len(PAIRS)} ordered links x {len(LAGS)} lags = {len(PAIRS) * len(LAGS)} tests"))


def run(x, y, lag, data, B, seed, suffix=None):
    dep, key, exog = link_design(NODES[x], NODES[y], CONTROLS, True, suffix or f"__lag{lag}")
    w = data[[dep] + exog].dropna()
    if len(w) < MIN_OBS or w.index.get_level_values(0).nunique() < 10:
        return None
    r = fe_wild_test(w, dep, key, exog, B=B, seed=seed)
    return {"coef": r["coef"], "t": r["t"], "p_cl": r["p_cl"], "p_wild": r["p_wild"], "n_obs": r["n_obs"],
            "n_countries": int(w.index.get_level_values(0).nunique())}


def bh_count(p, q):
    o = np.sort(p)
    ok_ = o <= q * (np.arange(1, len(o) + 1) / len(o))
    return int(np.max(np.where(ok_)[0]) + 1) if ok_.any() else 0


section("16-A — SCAN (216 tests)")
links = {}
for lag in LAGS:
    for x, y in PAIRS:
        r = run(x, y, lag, d0, B_BOOT, 42)
        if r is not None:
            links[f"{x}->{y}@{lag}"] = {"cause": x, "effect": y, "lag": lag, **r}
M = len(links)
keys = list(links)
p = np.array([links[k]["p_wild"] for k in keys])
order = np.argsort(p)
k_bh = bh_count(p, Q_FDR)
bh_ok = set(np.array(keys)[order[:k_bh]])
holm_ok = set()
for rank, idx in enumerate(order):
    if p[idx] <= 0.05 / (M - rank):
        holm_ok.add(keys[idx])
    else:
        break
for k in keys:
    links[k]["bh_survives"], links[k]["holm_survives"] = k in bh_ok, k in holm_ok
by_lag = {L: {"n": sum(1 for v in links.values() if v["lag"] == L),
              "raw_sig": sum(1 for v in links.values() if v["lag"] == L and v["p_wild"] < 0.05),
              "bh": sum(1 for v in links.values() if v["lag"] == L and v["bh_survives"])} for L in LAGS}
n_raw = int((p < 0.05).sum())
print(f"    raw p < 0.05: {n_raw} of {M} (chance {0.05 * M:.1f}) | BH survivors {k_bh} | Holm {len(holm_ok)}")
for L in LAGS:
    print(f"    lag {L}: {by_lag[L]}")

section("16-B — PLACEBO SCANS")
rng = np.random.default_rng(17)
pl_raw, pl_bh = [], []
for s in range(N_PLACEBO):
    d = d0.copy()
    for col in NODES.values():
        for L in LAGS:
            d[f"{col}__plc{L}"] = d.groupby(level=0)[f"{col}__lag{L}"].transform(lambda v: rng.permutation(v.to_numpy()))
    pp = []
    for lag in LAGS:
        for x, y in PAIRS:
            if f"{x}->{y}@{lag}" in links:
                r = run(x, y, lag, d, B_PLACEBO, s, suffix=f"__plc{lag}")
                pp.append(r["p_wild"] if r else 1.0)
    pp = np.array(pp)
    pl_raw.append(float((pp < 0.05).mean()))
    pl_bh.append(bh_count(pp, Q_FDR))
print(f"    placebo false-positive rate at raw 5%: mean {np.mean(pl_raw):.3f} (max {np.max(pl_raw):.3f}); BH survivors per placebo scan: mean {np.mean(pl_bh):.2f}, max {np.max(pl_bh)}")

section("16-C — SURVIVORS, THE PAPER'S LINKS, STRONGEST UNCORRECTED LINKS")
surv = sorted([k for k in keys if links[k]["bh_survives"]], key=lambda k: links[k]["p_wild"])
for k in surv:
    v = links[k]
    print(f"    {k:<30} coef {v['coef']:+.3f} | p {v['p_wild']:.3f} | Holm {'yes' if v['holm_survives'] else 'no'} | n {v['n_obs']}")
if not surv:
    print(warn("No link survives the false-discovery control across 216 tests."))
paper = ["violence->institutions", "institutions->FDI", "FDI->growth", "violence->FDI", "violence->growth", "institutions->growth"]
paper_rows = {k: {L: links.get(f"{k}@{L}") for L in LAGS} for k in paper}
for k in paper:
    print(f"    {k:<24} " + " | ".join(f"lag{L}: {paper_rows[k][L]['coef']:+.3f} (p {paper_rows[k][L]['p_wild']:.3f})" for L in LAGS if paper_rows[k][L]))
top = sorted(keys, key=lambda k: links[k]["p_wild"])[:12]
print("    12 smallest raw p-values:")
for k in top:
    v = links[k]
    print(f"      {k:<30} coef {v['coef']:+.3f} | p {v['p_wild']:.3f} | BH {'yes' if v['bh_survives'] else 'no'}")

section("16-C2 — RESOLUTION CHECK: the 15 smallest p-values re-estimated with 20,000 bootstrap draws")
# With 999 draws the smallest attainable p is 0.001, above the BH threshold of the top ranks (q/216 = 0.00046), so those links could not
# pass whatever their strength. The leaders are therefore re-estimated at finer resolution and the BH step is repeated.
B_REFINE, N_REFINE = 20000, 15
refined = {}
for k in top[:N_REFINE]:
    v = links[k]
    dep, key, exog = link_design(NODES[v["cause"]], NODES[v["effect"]], CONTROLS, True, f"__lag{v['lag']}")
    w = d0[[dep] + exog].dropna()
    refined[k] = float(fe_wild_test(w, dep, key, exog, B=B_REFINE, seed=5)["p_wild"])
p_ref = np.array([refined.get(k, links[k]["p_wild"]) for k in keys])
o_ref = np.argsort(p_ref)
thr = Q_FDR * np.arange(1, M + 1) / M
pass_ref = p_ref[o_ref] <= thr
k_bh_ref = int(np.max(np.where(pass_ref)[0]) + 1) if pass_ref.any() else 0
refined_rows = [{"link": keys[i], "coef": links[keys[i]]["coef"], "p_refined": float(p_ref[i]), "bh_threshold": float(thr[r])}
                for r, i in enumerate(o_ref[:N_REFINE])]
for r in refined_rows[:5]:
    print(f"    {r['link']:<30} coef {r['coef']:+.3f} | p {r['p_refined']:.5f} | BH threshold {r['bh_threshold']:.5f}")
print(f"    BH survivors after refinement: {k_bh_ref}")

section("16-D — CHAINS")
sv = [(links[k]["cause"], links[k]["effect"], links[k]["lag"]) for k in surv]
chains = [(a, b, la, c, lb) for (a, b, la) in sv for (b2, c, lb) in sv if b2 == b and c != a]
chain_res = []
cl = d0.reset_index()
for (x, m, la, y, lb) in chains:
    def ind(data, x=x, m=m, la=la, y=y, lb=lb):
        a_ = run(x, m, la, data, 1, 0); b_ = run(m, y, lb, data, 1, 0)
        return None if (a_ is None or b_ is None) else a_["coef"] * b_["coef"]
    obs = ind(d0)
    draws = country_cluster_bootstrap(cl, ind, B=B_CHAIN, seed=11)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    chain_res.append({"chain": f"{x} -({la})-> {m} -({lb})-> {y}", "indirect": obs, "ci_lo": float(lo), "ci_hi": float(hi),
                      "excludes_zero": bool(lo > 0 or hi < 0)})
    print(f"    {chain_res[-1]['chain']}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]")
if not chains:
    print(warn("No pair of surviving links composes into a chain."))

section("16-E — FIGURE")
names = list(NODES)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
for ax, L in zip(axes, LAGS):
    mat = np.full((9, 9), np.nan)
    for k, v in links.items():
        if v["lag"] == L:
            i, j = names.index(v["cause"]), names.index(v["effect"])
            mat[i, j] = np.sign(v["coef"]) * -np.log10(max(v["p_wild"], 1e-3))
            if v["bh_survives"]:
                ax.text(j, i, "**", ha="center", va="center", fontsize=13)
            elif v["p_wild"] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=13)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(9)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(names if L == 1 else [], fontsize=8)
    ax.set_title(f"cause lagged {L} year{'s' if L > 1 else ''}", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.02).set_label("sign(coef) x -log10(p, wild bootstrap)")
fig.suptitle(f"Lagged links, {M} tests in one family  (** survives BH q={Q_FDR}, * raw p<0.05)", fontsize=11)
fig.savefig(DIRS["figures"] / "21_link_scan_lags.png", dpi=250, bbox_inches="tight")
plt.close(fig)

save_json({
    "nodes": NODES, "lags": list(LAGS), "n_tests": M, "links": links, "by_lag": {str(k): v for k, v in by_lag.items()},
    "paper_links": {k: {str(L): v for L, v in d.items()} for k, d in paper_rows.items()},
    "summary": {"n_raw_sig_05": n_raw, "expected_by_chance": 0.05 * M, "n_bh_survivors": k_bh, "n_holm_survivors": len(holm_ok), "fdr_q": Q_FDR,
                "placebo_false_positive_rate_mean": float(np.mean(pl_raw)), "placebo_false_positive_rate_max": float(np.max(pl_raw)),
                "placebo_bh_survivors_mean": float(np.mean(pl_bh)), "placebo_bh_survivors_max": int(np.max(pl_bh)),
                "n_placebo_scans": N_PLACEBO, "n_bootstrap": B_BOOT},
    "refined": {"n_draws": B_REFINE, "rows": refined_rows, "n_bh_survivors_refined": k_bh_ref},
    "chains": chain_res,
}, DIRS["json"] / "16_link_scan_lags.json")
print(bold(f"\nSummary: {n_raw}/{M} raw-significant (chance {0.05 * M:.1f}) | {k_bh} survive BH | {len(chain_res)} chains"))
