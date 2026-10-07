"""
11_spec_curve.py
════════════════════════════════════════════════════════════════════════
Layer 11 — Specification Curve for the Institution Index

MOTIVATION
──────────
Module 01/05 and the manuscript report that moving the institution index
from 3 to 6 Worldwide Governance Indicators (WGI) dimensions changes
which links of the violence → institutions → FDI → growth chain are
significant. That is a comparison of TWO specifications. This module
runs ALL of them: every subset of two or more of the six dimensions
(57 indices) plus each single dimension (6), giving 63 institution
indices, and re-estimates the two index-dependent links for each:

    EQ1:  index_S  ~ homicide_rate_log_lag1 + controls   (violence → institutions)
    EQ2:  FDI      ~ index_S               + controls   (institutions → FDI)

all with the same Two-Way Fixed Effects specification as Module 02.
The result is the full distribution of estimates a researcher could
have reported from this one data set depending on an arbitrary index
construction choice -- a specification curve (Simonsohn, Simmons &
Nelson, 2020; see also Brodeur, Cook & Heyes, 2020 on specification
searching).

For each subset S the index is the equal-weighted mean of the z-scored
dimensions in S (same standardization as Module 01), defined on
country-years where every dimension in S is observed.

Two standard errors are recorded for every specification: clustered by
country (conventional; anti-conservative with few clusters) and
Driscoll-Kraay. Neither is the wild-cluster bootstrap used for the
paper's headline numbers (too slow for 126 fits); the specification
curve is meant to show how the POINT ESTIMATE and its nominal
significance move with the index definition, not to replace the
headline inference.

Reference
──────────────────────────────────────────────────────────────────────
Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification
curve analysis. Nature Human Behaviour, 4, 1208-1214.
Brodeur, A., Cook, N., & Heyes, A. (2020). Methods Matter: p-Hacking and
Publication Bias in Causal Analysis in Economics. AER, 110(11), 3634-3660.
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, bold, save_json,
    make_output_dirs, build_multiindex, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
BLUE, RED_C, GREY, AMBER = "#2563EB", "#DC2626", "#9CA3AF", "#D97706"

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("11-A — LOAD")
df = pd.read_csv(Path(__file__).parent / "panel_enriched.csv")
with open(Path(__file__).parent / "json" / "01_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
DIMS = meta["inst_vars"]
CONTROLS = [c for c in meta["controls"] if c != "year_c"]
ORIGINAL_3 = ["rule_of_law", "control_corruption", "political_stability"]
KEY = "homicide_rate_log_lag1"
SHORT = {"rule_of_law": "RL", "control_corruption": "CC", "political_stability": "PS",
         "voice_accountability": "VA", "government_effectiveness": "GE", "regulatory_quality": "RQ"}
print(ok(f"Panel: {len(df)} obs | {df[ENTITY_COL].nunique()} countries | {len(DIMS)} WGI dimensions"))

subsets = [s for k in range(1, len(DIMS) + 1) for s in itertools.combinations(DIMS, k)]
print(ok(f"{len(subsets)} institution indices (6 single dimensions + {len(subsets) - len(DIMS)} composites)"))


def fit(dep_df, dep, exog, key):
    work = dep_df[[dep] + exog].dropna()
    if len(work) < len(exog) + 30:
        return None
    mod = PanelOLS(work[dep], work[exog], entity_effects=True, time_effects=True, drop_absorbed=True)
    r_cl = mod.fit(cov_type="clustered", cluster_entity=True)
    t_bar = work.reset_index()[TIME_COL].nunique()
    bw = max(1, int(np.floor(4 * (t_bar / 100) ** (2 / 9))))
    r_dk = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    return {
        "coef": float(r_cl.params[key]), "se_cl": float(r_cl.std_errors[key]),
        "p_cl": float(r_cl.pvalues[key]), "p_dk": float(r_dk.pvalues[key]),
        "ci_lo": float(r_cl.conf_int().loc[key].iloc[0]), "ci_hi": float(r_cl.conf_int().loc[key].iloc[1]),
        "n_obs": int(r_cl.nobs),
    }


# ════════════════════════════════════════════════════════════════════════
# ESTIMATE ALL SPECIFICATIONS
# ════════════════════════════════════════════════════════════════════════
section("11-B — ESTIMATE EQ1 AND EQ2 FOR EVERY INDEX DEFINITION")
rows = []
for S in subsets:
    zcols = [f"{d}_z" for d in S]
    d = df.copy()
    d["idx"] = d[zcols].mean(axis=1).where(d[zcols].notna().all(axis=1))
    d = build_multiindex(d)
    e1 = fit(d, "idx", [KEY] + CONTROLS, KEY)
    e2 = fit(d, "fdi_percent_gdp", ["idx"] + CONTROLS, "idx")
    if e1 is None or e2 is None:
        continue
    rows.append({
        "dims": list(S), "label": "+".join(SHORT[x] for x in S), "k": len(S),
        "is_primary_6": len(S) == 6, "is_original_3": sorted(S) == sorted(ORIGINAL_3),
        "EQ1": e1, "EQ2": e2,
    })
print(ok(f"{len(rows)} specifications estimated"))


def summarize(eq, expected_sign):
    sel = [r for r in rows]
    coefs = np.array([r[eq]["coef"] for r in sel])
    p_cl = np.array([r[eq]["p_cl"] for r in sel])
    p_dk = np.array([r[eq]["p_dk"] for r in sel])
    right = np.sign(coefs) == expected_sign
    prim = next(r for r in rows if r["is_primary_6"])[eq]
    orig = next(r for r in rows if r["is_original_3"])[eq]
    return {
        "n_specs": len(sel),
        "expected_sign": "negative" if expected_sign < 0 else "positive",
        "share_expected_sign": float(right.mean()),
        "share_sig_clustered_05": float((p_cl < 0.05).mean()),
        "share_sig_clustered_05_and_expected_sign": float(((p_cl < 0.05) & right).mean()),
        "share_sig_dk_05": float((p_dk < 0.05).mean()),
        "share_sig_both_05": float(((p_cl < 0.05) & (p_dk < 0.05)).mean()),
        "coef_min": float(coefs.min()), "coef_median": float(np.median(coefs)), "coef_max": float(coefs.max()),
        "primary_6dim": {"coef": prim["coef"], "p_cl": prim["p_cl"], "p_dk": prim["p_dk"]},
        "original_3dim": {"coef": orig["coef"], "p_cl": orig["p_cl"], "p_dk": orig["p_dk"]},
    }


summary = {"EQ1": summarize("EQ1", -1), "EQ2": summarize("EQ2", +1)}
for eq, lab in (("EQ1", "violence → institutions"), ("EQ2", "institutions → FDI")):
    s = summary[eq]
    subsection(f"{eq} ({lab}) across {s['n_specs']} index definitions")
    print(f"    coefficient: min {s['coef_min']:+.3f} | median {s['coef_median']:+.3f} | max {s['coef_max']:+.3f}")
    print(f"    share with the theoretically expected ({s['expected_sign']}) sign: {s['share_expected_sign']:.0%}")
    print(f"    share significant at 5%: clustered {s['share_sig_clustered_05']:.0%} | Driscoll-Kraay "
          f"{s['share_sig_dk_05']:.0%} | both {s['share_sig_both_05']:.0%}")
    print(f"    primary (6-dim): coef {s['primary_6dim']['coef']:+.3f}, p_cl {s['primary_6dim']['p_cl']:.3f}")
    print(f"    original (3-dim): coef {s['original_3dim']['coef']:+.3f}, p_cl {s['original_3dim']['p_cl']:.3f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("11-C — SPECIFICATION CURVE FIGURE")
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1.4], hspace=0.08, wspace=0.18)
for col, (eq, title) in enumerate((("EQ1", "EQ1: violence → institution index"), ("EQ2", "EQ2: institution index → FDI (% GDP)"))):
    order = sorted(range(len(rows)), key=lambda i: rows[i][eq]["coef"])
    ax = fig.add_subplot(gs[0, col])
    for pos, i in enumerate(order):
        r = rows[i][eq]
        sig = r["p_cl"] < 0.05
        c = BLUE if sig else GREY
        ax.vlines(pos, r["ci_lo"], r["ci_hi"], color=c, alpha=0.35, linewidth=1.2)
        ax.plot(pos, r["coef"], "o", color=c, markersize=3.5)
        if rows[i]["is_primary_6"]:
            ax.plot(pos, r["coef"], "o", color=RED_C, markersize=9, mfc="none", mew=2)
        if rows[i]["is_original_3"]:
            ax.plot(pos, r["coef"], "s", color=AMBER, markersize=9, mfc="none", mew=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Coefficient (95% CI, clustered)" if col == 0 else "")
    ax.set_xticks([])
    ax.plot([], [], "o", color=BLUE, label="p<0.05 (clustered)")
    ax.plot([], [], "o", color=GREY, label="n.s.")
    ax.plot([], [], "o", color=RED_C, mfc="none", mew=2, markersize=8, label="primary: all 6 dims")
    ax.plot([], [], "s", color=AMBER, mfc="none", mew=2, markersize=8, label="original: RL+CC+PS")
    if col == 0:
        ax.legend(fontsize=7.5, loc="best")
    axm = fig.add_subplot(gs[1, col], sharex=ax)
    for pos, i in enumerate(order):
        for yi, d in enumerate(DIMS):
            inc = d in rows[i]["dims"]
            axm.plot(pos, yi, "s", color="black" if inc else "#E5E7EB", markersize=3.2)
    axm.set_yticks(range(len(DIMS)))
    axm.set_yticklabels([SHORT[d] for d in DIMS], fontsize=8)
    axm.set_xlabel("Specifications, sorted by coefficient" if True else "")
    axm.set_xticks([])
    axm.grid(False)
fig.suptitle("Specification curve: every subset of the six WGI dimensions as the institution index",
             fontsize=12, fontweight="bold")
fig.savefig(DIRS["figures"] / "16_spec_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/16_spec_curve.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("11-D — EXPORT")
save_json({"specs": rows, "summary": summary, "controls": CONTROLS, "dims": DIMS},
          DIRS["json"] / "11_spec_curve.json")
print(f"\n{bold('Module 11 complete.')}")
