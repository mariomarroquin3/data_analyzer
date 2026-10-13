"""
13_causal_link_scan.py
════════════════════════════════════════════════════════════════════════
Layer 13 — A Disciplined Scan of Candidate Causal Links and Chains

QUESTION
────────
The paper's chain is violence → institutions → FDI → growth. Which OTHER
links between the panel's variables carry statistical evidence, and do any
of them compose into chains? Scanning many links invites false positives,
so the scan is built to limit them:

  - The family of tests is fixed in advance: every ordered pair among nine
    variables (72 links), not only the ones that turn out to look good.
  - Temporal precedence: the putative cause enters lagged one year.
  - Primary specification (Granger-style): the outcome's own lag is a control,
    so a link must predict CHANGES in the outcome, not just its persistent
    level (the main reason cross-country links look spuriously strong).
    A specification without the lagged outcome is reported alongside.
  - Country and year fixed effects; wild cluster bootstrap p-values (G = 18).
  - Multiplicity: Benjamini-Hochberg false-discovery control (q = 0.10)
    across the 72 primary tests, plus Holm family-wise control.
  - Calibration: the same scan is re-run 20 times with each cause's years
    shuffled within country (destroying any real link, preserving its
    distribution); the share of "significant" links in those placebo scans
    is the procedure's own false-positive rate.
  - Direction: for each pair both directions are tested, so a link whose
    reverse direction is as strong is flagged as ambiguous rather than causal.
  - Chains X → M → Y are examined only when BOTH links survive; the indirect
    effect gets a country-cluster bootstrap interval.

This is observational, lagged-association evidence. Surviving the filters
means "a lagged association that is not explained by country/year effects
or by the outcome's own persistence" -- not identification of a causal
effect.

References
──────────────────────────────────────────────────────────────────────
Granger, C. W. J. (1969). Econometrica, 37(3), 424-438.
Benjamini, Y., & Hochberg, Y. (1995). JRSS-B, 57(1), 289-300.
Holm, S. (1979). Scandinavian Journal of Statistics, 6(2), 65-70.
Nickell, S. (1981). Biases in dynamic models with fixed effects.
  Econometrica, 49(6), 1417-1426. (Lagged-outcome FE bias is O(1/T);
  T is about 24 here.)
Dumitrescu, E.-I., & Hurlin, C. (2012). Testing for Granger non-causality
  in heterogeneous panels. Economic Modelling, 29(4), 1450-1460.
════════════════════════════════════════════════════════════════════════
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
    set_plot_style, section, subsection, ok, warn, bold, save_json,
    make_output_dirs, build_multiindex, fe_wild_test, country_cluster_bootstrap, link_design, ENTITY_COL, TIME_COL,
)

set_plot_style()
HERE = Path(__file__).parent
DIRS = make_output_dirs(HERE)
BLUE, RED_C, GREY, AMBER = "#2563EB", "#DC2626", "#9CA3AF", "#D97706"
B_BOOT, B_PLACEBO, N_PLACEBO, B_CHAIN = 999, 199, 20, 499
Q_FDR, MIN_OBS = 0.10, 150

section("13-A — LOAD")
df = pd.read_csv(HERE / "panel_enriched.csv")
with open(HERE / "json" / "01_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
CONTROLS = [c for c in meta["controls"] if c != "year_c"]

NODES = {  # short label -> column
    "violence": "homicide_rate_log",
    "institutions": "inst_avg",
    "FDI": "fdi_percent_gdp",
    "growth": "gdp_growth",
    "inflation": "inflation",
    "unemployment": "unemployment",
    "exports": "exports_percent_gdp",
    "tourism": "tourist_arrivals_log",
    "remittances": "remittances_percent_gdp",
}
d0 = build_multiindex(df)
for col in NODES.values():
    d0[f"{col}__lag1"] = d0.groupby(level=0)[col].shift(1)
print(ok(f"{len(NODES)} variables -> {len(NODES) * (len(NODES) - 1)} ordered links; {df[ENTITY_COL].nunique()} countries"))


def design(x, y, lag_y=True, cause_suffix="__lag1"):
    """Dependent column, cause column, and regressor list for the link x -> y.

    The cause is `<x><cause_suffix>` (the lagged series, or its shuffled placebo
    copy); the lagged-outcome control is always the REAL `<y>__lag1`.
    """
    return link_design(NODES[x], NODES[y], CONTROLS, lag_y, cause_suffix)


def run_link(x, y, data, lag_y=True, B=B_BOOT, seed=42, cause_suffix="__lag1"):
    dep, key, exog = design(x, y, lag_y, cause_suffix)
    w = data[[dep] + exog].dropna()
    if len(w) < MIN_OBS or w.index.get_level_values(0).nunique() < 10:
        return None
    r = fe_wild_test(w, dep, key, exog, B=B, seed=seed)
    return {"coef": r["coef"], "t": r["t"], "p_cl": r["p_cl"], "p_wild": r["p_wild"], "n_obs": r["n_obs"],
            "n_countries": int(w.index.get_level_values(0).nunique())}


# ════════════════════════════════════════════════════════════════════════
# SCAN
# ════════════════════════════════════════════════════════════════════════
section("13-B — SCAN ALL ORDERED LINKS (primary: lagged outcome control; alt: without)")
links = {}
for x, y in itertools.permutations(NODES, 2):
    a = run_link(x, y, d0, lag_y=True)
    b = run_link(x, y, d0, lag_y=False)
    if a is None:
        continue
    links[f"{x}->{y}"] = {"cause": x, "effect": y, "primary": a, "no_lagged_outcome": b}
M = len(links)
print(ok(f"{M} links estimated"))

# multiplicity control on the primary p-values
keys = list(links)
p = np.array([links[k]["primary"]["p_wild"] for k in keys])
order = np.argsort(p)
bh_thresh = Q_FDR * (np.arange(1, M + 1) / M)
passed = p[order] <= bh_thresh
k_bh = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
bh_ok = set(np.array(keys)[order[:k_bh]])
holm_ok = set()
for rank, idx in enumerate(order):
    if p[idx] <= 0.05 / (M - rank):
        holm_ok.add(keys[idx])
    else:
        break
for k in keys:
    L = links[k]
    L["bh_survives"] = k in bh_ok
    L["holm_survives"] = k in holm_ok
    L["alt_agrees"] = bool(L["no_lagged_outcome"] and np.sign(L["no_lagged_outcome"]["coef"]) == np.sign(L["primary"]["coef"])
                           and L["no_lagged_outcome"]["p_wild"] < 0.10)
    rev = links.get(f"{L['effect']}->{L['cause']}")
    L["reverse_p_wild"] = rev["primary"]["p_wild"] if rev else None

n_raw = int((p < 0.05).sum())
print(f"    raw p_wild < 0.05: {n_raw} of {M} (about {0.05 * M:.1f} expected by chance if no link were real)")
print(f"    survive Benjamini-Hochberg (q = {Q_FDR}): {k_bh} | survive Holm (5%): {len(holm_ok)}")

# ════════════════════════════════════════════════════════════════════════
# PLACEBO CALIBRATION
# ════════════════════════════════════════════════════════════════════════
section("13-C — PLACEBO SCANS (cause shuffled over years within country)")
rng = np.random.default_rng(7)
placebo_raw, placebo_bh = [], []
for s in range(N_PLACEBO):
    d = d0.copy()
    for col in NODES.values():
        # Shuffled COPY of each cause (within country: keeps its distribution, breaks its timing).
        # The real `<col>__lag1` stays untouched because it is the lagged-outcome control.
        d[f"{col}__plc"] = d.groupby(level=0)[f"{col}__lag1"].transform(lambda v: rng.permutation(v.to_numpy()))
    pp = []
    for x, y in itertools.permutations(NODES, 2):
        if f"{x}->{y}" not in links:
            continue
        r = run_link(x, y, d, lag_y=True, B=B_PLACEBO, seed=s, cause_suffix="__plc")
        pp.append(r["p_wild"] if r else 1.0)
    pp = np.array(pp)
    placebo_raw.append(float((pp < 0.05).mean()))
    o = np.sort(pp)
    ok_bh = o <= Q_FDR * (np.arange(1, len(o) + 1) / len(o))
    placebo_bh.append(int(np.max(np.where(ok_bh)[0]) + 1) if ok_bh.any() else 0)
print(f"    placebo false-positive rate at raw 5%: mean {np.mean(placebo_raw):.3f} (max {np.max(placebo_raw):.3f}); nominal 0.050")
print(f"    placebo links surviving BH: mean {np.mean(placebo_bh):.2f} per scan (max {np.max(placebo_bh)})")

# ════════════════════════════════════════════════════════════════════════
# SURVIVORS AND DIRECTION
# ════════════════════════════════════════════════════════════════════════
section("13-D — LINKS THAT SURVIVE")
surv = [k for k in keys if links[k]["bh_survives"]]
if not surv:
    print(warn("No link survives the false-discovery control."))
for k in sorted(surv, key=lambda k: links[k]["primary"]["p_wild"]):
    L = links[k]
    pr = L["primary"]
    rp = L["reverse_p_wild"]
    flag = "bidirectional" if rp is not None and rp < 0.10 else "one-directional"
    print(f"    {k:<28} coef {pr['coef']:+.3f} | p_wild {pr['p_wild']:.3f} | Holm {'yes' if L['holm_survives'] else 'no '} | "
          f"alt spec agrees {'yes' if L['alt_agrees'] else 'no '} | reverse p {rp:.3f} ({flag}) | n {pr['n_obs']}")

# the paper's own chain, for reference
paper_links = ["violence->institutions", "institutions->FDI", "FDI->growth", "institutions->growth", "violence->growth", "violence->FDI"]
subsection("The paper's chain under this scan")
for k in paper_links:
    if k in links:
        pr = links[k]["primary"]
        print(f"    {k:<26} coef {pr['coef']:+.3f} | p_wild {pr['p_wild']:.3f} | BH {'yes' if links[k]['bh_survives'] else 'no'}")

# ════════════════════════════════════════════════════════════════════════
# CHAINS
# ════════════════════════════════════════════════════════════════════════
section("13-E — CHAINS X -> M -> Y WITH BOTH LINKS SURVIVING")
survivors = {(links[k]["cause"], links[k]["effect"]) for k in surv}
chains = [(x, m, y) for (x, m) in survivors for (m2, y) in survivors if m2 == m and y != x]
chain_res = []
cl = d0.reset_index()
for (x, m, y) in chains:
    def indirect(data, x=x, m=m, y=y):
        a = run_link(x, m, data, lag_y=True, B=1)
        b = run_link(m, y, data, lag_y=True, B=1)
        return None if (a is None or b is None) else a["coef"] * b["coef"]
    obs = indirect(d0)
    draws = country_cluster_bootstrap(cl, indirect, B=B_CHAIN, seed=11)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    chain_res.append({"chain": f"{x} -> {m} -> {y}", "indirect": obs, "ci_lo": float(lo), "ci_hi": float(hi),
                      "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(draws)})
    print(f"    {x} -> {m} -> {y}: indirect {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")
if not chains:
    print(warn("No pair of surviving links composes into a chain."))

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("13-F — LINK MAP FIGURE")
names = list(NODES)
mat = np.full((len(names), len(names)), np.nan)
ann = [["" for _ in names] for _ in names]
for k, L in links.items():
    i, j = names.index(L["cause"]), names.index(L["effect"])
    pr = L["primary"]
    mat[i, j] = np.sign(pr["coef"]) * -np.log10(max(pr["p_wild"], 1e-3))
    ann[i][j] = ("**" if L["bh_survives"] else "*" if pr["p_wild"] < 0.05 else "")
fig, ax = plt.subplots(figsize=(8.6, 7))
im = ax.imshow(mat, cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel("effect (outcome at t)"); ax.set_ylabel("cause (lagged one year)")
for i in range(len(names)):
    for j in range(len(names)):
        if ann[i][j]:
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=13, color="black")
        elif i == j:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#E5E7EB"))
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("sign(coef) × −log10(p, wild bootstrap)")
ax.set_title(f"Lagged links (two-way FE, lagged outcome control)\n** survives FDR q={Q_FDR}   * raw p<0.05 only", fontsize=10)
fig.tight_layout()
fig.savefig(DIRS["figures"] / "18_causal_link_map.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/18_causal_link_map.png"))

save_json({
    "nodes": NODES,
    "n_links": M,
    "links": links,
    "summary": {
        "n_raw_sig_05": n_raw, "expected_by_chance": 0.05 * M, "n_bh_survivors": k_bh, "n_holm_survivors": len(holm_ok),
        "fdr_q": Q_FDR, "placebo_false_positive_rate_mean": float(np.mean(placebo_raw)),
        "placebo_false_positive_rate_max": float(np.max(placebo_raw)),
        "placebo_bh_survivors_mean": float(np.mean(placebo_bh)), "placebo_bh_survivors_max": int(np.max(placebo_bh)),
        "n_placebo_scans": N_PLACEBO, "n_bootstrap": B_BOOT,
    },
    "chains": chain_res,
}, DIRS["json"] / "13_causal_link_scan.json")
print(bold(f"\nSummary: {n_raw}/{M} raw-significant (chance {0.05 * M:.1f}) | {k_bh} survive BH | {len(chain_res)} chains tested"))
