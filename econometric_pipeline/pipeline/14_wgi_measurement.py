"""
14_wgi_measurement.py
════════════════════════════════════════════════════════════════════════
Layer 14 — How Much Can the Institution Measure Be Trusted?

MOTIVATION
──────────
The six institution variables are World Bank Worldwide Governance
Indicators (WGI): perception-based aggregates of 35 expert assessments and
household / firm surveys, combined with an Unobserved Components Model that
reports a margin of error with every estimate (Kaufmann & Kraay, 2024; Jiang
et al., 2025). The 0-100 score used here is a linear rescaling of that
estimate between two fixed hypothetical benchmark countries, not a percentile.
Earlier modules treat the score as exact. This module uses the WGI's own
90% intervals (foundational_datasets/wgi_confidence_intervals.csv, fetched by
data_extraction/WGI_ci_get.py) to ask five questions:

  1. How large is the measurement noise relative to the variation the
     fixed-effects models exploit (within-country, year to year)?
  2. Which observed changes -- one, five and ten years apart, and El Salvador's
     2020-2024 drop -- are distinguishable from that noise?
  3. How much could classical measurement error attenuate the coefficient
     when the index is a REGRESSOR (EQ2)?
  4. Does the violence -> institutions link (EQ1) live in the dimensions whose
     sources contain crime / violence questions (Rule of Law; Political
     Stability) and not in the others? That is what a mechanical overlap would
     look like. The grouping was chosen AFTER seeing the per-dimension
     decomposition of EQ1 (README), so it is a post-hoc, exploratory
     classification, not a pre-specified test; Government Effectiveness, whose
     sources were not audited, shows the link too, so overlap is at most one of
     several possible explanations (a real erosion of state capacity is another).
  5. Are the main results sensitive to country-years supported by few sources?

IMPORTANT LIMITS
────────────────
- Independence: the published standard errors describe the uncertainty of each
  estimate separately. Errors of the same country in adjacent years, and of
  different dimensions fed by the same source, are probably positively
  correlated. Treating them as independent OVERSTATES the noise in changes and
  in within-country variation, so the reliability and "distinguishable change"
  figures are conservative (worst-case) bounds, not point estimates.
- Content of the dimensions: the Rule of Law source list (WGI 2025) includes
  violent crime, victimisation and assault items; Political Stability is
  defined by the likelihood of politically motivated violence. The source lists
  of the other four dimensions were not audited here, so "no crime content" is
  taken from their definitions and should be read as such.
- Classical measurement error biases a regressor's coefficient toward zero by
  the reliability factor but does not change its t-statistic (the standard
  error is inflated in the same proportion); it therefore cannot by itself
  explain a non-significant link.

References
──────────────────────────────────────────────────────────────────────
Kaufmann, D., & Kraay, A. (2024). The Worldwide Governance Indicators:
  Methodology and 2024 Update. World Bank.
Jiang, C. N., Jain, A., Romano, T., & Boyce, T. (2025). The Worldwide
  Governance Indicators: Revised Methodology for Measuring Governance Using
  Perception Data. World Bank.
Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel
  Data, 2nd ed., ch. 4 (measurement error).
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

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, bold, save_json,
    make_output_dirs, build_multiindex, fe_wild_test, ENTITY_COL, TIME_COL,
)

set_plot_style()
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DIRS = make_output_dirs(HERE)
BLUE, RED_C, GREY, AMBER, GREEN = "#2563EB", "#DC2626", "#9CA3AF", "#D97706", "#059669"
Z90 = 1.645
B_BOOT = 999

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("14-A — LOAD")
panel = pd.read_csv(HERE / "panel_enriched.csv")
with open(HERE / "json" / "01_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
DIMS = meta["inst_vars"]
CONTROLS = [c for c in meta["controls"] if c != "year_c"]
ci = pd.read_csv(ROOT / "foundational_datasets" / "wgi_confidence_intervals.csv")
ci["se"] = (ci["ub"] - ci["lb"]) / (2 * Z90)          # implied standard error on the 0-100 scale
ci["width"] = ci["ub"] - ci["lb"]

# the downloaded scores must be the ones the panel was built from
chk = ci.pivot_table(index=[ENTITY_COL, TIME_COL], columns="dim", values="score").reset_index()
mrg = panel.merge(chk, on=[ENTITY_COL, TIME_COL], suffixes=("", "_api"))
max_gap = max(float((mrg[d] - mrg[d + "_api"]).abs().max()) for d in DIMS)
assert max_gap < 1e-4, f"panel scores differ from the WGI API download (max gap {max_gap})"
print(ok(f"{len(ci)} country-dimension-years | panel scores match the download (max gap {max_gap:.1e})"))

# ════════════════════════════════════════════════════════════════════════
# 1. NOISE VS WITHIN-COUNTRY VARIATION
# ════════════════════════════════════════════════════════════════════════
section("14-B — SIZE OF THE 90% INTERVALS")
sd_pooled = {d: float(panel[d].std()) for d in DIMS}
sd_within = {d: float((panel[d] - panel.groupby(ENTITY_COL)[d].transform("mean")).std()) for d in DIMS}
noise_tab = {}
for d in DIMS:
    s = ci[ci["dim"] == d]
    noise_tab[d] = {
        "mean_ci_width": float(s["width"].mean()), "mean_se": float(s["se"].mean()),
        "mean_sources": float(s["n_sources"].mean()), "min_sources": int(s["n_sources"].min()),
        "sd_pooled": sd_pooled[d], "sd_within_country": sd_within[d],
        "noise_to_within_sd": float(s["se"].mean() / sd_within[d]),
    }
    t = noise_tab[d]
    print(f"    {d:<26} 90% CI width {t['mean_ci_width']:5.1f} | SE {t['mean_se']:4.2f} | within-country SD {t['sd_within_country']:4.2f} | "
          f"ratio {t['noise_to_within_sd']:.2f} | sources {t['mean_sources']:.1f} (min {t['min_sources']})")

# ════════════════════════════════════════════════════════════════════════
# 2. DISTINGUISHABLE CHANGES
# ════════════════════════════════════════════════════════════════════════
section("14-C — WHICH CHANGES ARE DISTINGUISHABLE FROM NOISE?")
change_share = {}
for d in DIMS:
    s = ci[ci["dim"] == d].set_index([ENTITY_COL, TIME_COL])
    a = s["score"].unstack(ENTITY_COL)
    e = s["se"].unstack(ENTITY_COL)
    change_share[d] = {}
    for lag in (1, 5, 10):
        # the WGI has no 2001 estimate, so shift by calendar years on the year index
        prev = a.reindex(a.index - lag)
        prev.index = a.index
        eprev = e.reindex(e.index - lag)
        eprev.index = e.index
        delta = (a - prev).dropna(how="all")
        se = np.sqrt(e ** 2 + eprev ** 2).reindex(delta.index)
        sig = (delta.abs() > Z90 * se).where(delta.notna())
        change_share[d][str(lag)] = {"share_significant": float(sig.stack().mean()), "n": int(sig.stack().size)}
    print(f"    {d:<26} year-on-year {change_share[d]['1']['share_significant']:.1%} | 5 years {change_share[d]['5']['share_significant']:.1%} | "
          f"10 years {change_share[d]['10']['share_significant']:.1%}")

subsection("Changes 2020 -> 2024, all countries")
START, END = 2020, 2024
changes = []
for c in sorted(panel[ENTITY_COL].unique()):
    for d in DIMS:
        s = ci[(ci["dim"] == d) & (ci[ENTITY_COL] == c)].set_index(TIME_COL)
        if START in s.index and END in s.index:
            d0, d1 = s.loc[START], s.loc[END]
            delta = float(d1["score"] - d0["score"])
            margin = float(Z90 * np.hypot(d0["se"], d1["se"]))
            changes.append({"country": c, "dim": d, "start": float(d0["score"]), "end": float(d1["score"]), "delta": delta,
                            "margin90": margin, "distinguishable": bool(abs(delta) > margin),
                            "sources_start": int(d0["n_sources"]), "sources_end": int(d1["n_sources"])})
chg = pd.DataFrame(changes)
slv = chg[chg["country"] == "SLV"]
for _, r in slv.iterrows():
    print(f"    SLV {r['dim']:<26} {r['start']:5.1f} -> {r['end']:5.1f} ({r['delta']:+5.1f}, margin +/-{r['margin90']:4.1f}) "
          f"{'distinguishable' if r['distinguishable'] else 'within noise'} | sources {r['sources_start']}->{r['sources_end']}")
n_dist = int(chg["distinguishable"].sum())
print(ok(f"{n_dist} of {len(chg)} country-dimension changes 2020-2024 are distinguishable from noise "
         f"({int((chg['distinguishable'] & (chg['country'] != 'SLV')).sum())} outside El Salvador)"))

# ════════════════════════════════════════════════════════════════════════
# 3. RELIABILITY / ATTENUATION (WORST CASE)
# ════════════════════════════════════════════════════════════════════════
section("14-D — RELIABILITY OF THE INDEX INSIDE THE FIXED-EFFECTS REGRESSION (worst case)")
se_long = ci.pivot_table(index=[ENTITY_COL, TIME_COL], columns="dim", values="se").reset_index()
mm = panel.merge(se_long, on=[ENTITY_COL, TIME_COL], suffixes=("", "_se"))
for d in DIMS:
    mm[d + "_sez"] = mm[d + "_se"] / sd_pooled[d]            # standard error in z units
sez = mm[[d + "_sez" for d in DIMS]].to_numpy()
mm["noise_var_index"] = (sez ** 2).sum(axis=1) / len(DIMS) ** 2    # errors independent across dimensions

# stack of (dependent, key, noise variance column)
def within_reliability(data, dep, key, noise_col):
    """beta on `key` (two-way FE + controls) and 1 - noise / residual variance of key (Q = residual maker)."""
    cols = [dep, key] + CONTROLS
    w = data.dropna(subset=cols).copy()
    ent = pd.get_dummies(w[ENTITY_COL]).to_numpy(float)
    yr = pd.get_dummies(w[TIME_COL]).to_numpy(float)[:, 1:]
    Z = np.hstack([w[CONTROLS].to_numpy(float), ent, yr])
    Q = np.eye(len(w)) - Z @ np.linalg.pinv(Z)
    xt = Q @ w[key].to_numpy(float)
    beta = float(xt @ (Q @ w[dep].to_numpy(float)) / (xt @ xt))
    noise = float((np.diag(Q) * w[noise_col].to_numpy(float)).sum())
    return beta, 1.0 - noise / float(xt @ xt), len(w)


beta_idx, lam_idx, n_idx = within_reliability(mm, "fdi_percent_gdp", "inst_avg", "noise_var_index")
reliability = {"index": {"beta_eq2": beta_idx, "reliability": lam_idx, "beta_if_reliability": beta_idx / lam_idx, "n_obs": n_idx}}
print(f"    index (EQ2 sample, n={n_idx}): observed beta {beta_idx:+.3f} | worst-case reliability {lam_idx:.3f} | "
      f"beta if that reliability held: {beta_idx / lam_idx:+.3f}")
for d in DIMS:
    mm["_nv"] = mm[d + "_sez"] ** 2
    mm["_z"] = (mm[d] - panel[d].mean()) / sd_pooled[d]
    b, lam, n = within_reliability(mm, "fdi_percent_gdp", "_z", "_nv")
    reliability[d] = {"beta_eq2": b, "reliability": lam, "n_obs": n}
    print(f"    {d:<26} worst-case reliability {lam:.3f}")
print(warn("Errors of the same country in adjacent years are probably positively correlated, so these reliabilities are lower bounds; "
           "values near or below zero for single dimensions mean the independence assumption is too pessimistic to be informative there."))

# ════════════════════════════════════════════════════════════════════════
# 4. EQ1 BY DIMENSION CONTENT
# ════════════════════════════════════════════════════════════════════════
section("14-E — EQ1 AND EQ2 BY WHETHER THE DIMENSION CONTAINS CRIME / VIOLENCE ITEMS")
GROUPS = {
    "violence_related": ["rule_of_law", "political_stability"],
    "no_crime_content": ["control_corruption", "voice_accountability", "regulatory_quality"],
    "government_effectiveness": ["government_effectiveness"],
    "all_six": list(DIMS),
}
by_content = {}
for name, dims in GROUPS.items():
    zc = [f"{d}_z" for d in dims]
    d_ = panel.copy()
    d_["idx"] = d_[zc].mean(axis=1).where(d_[zc].notna().all(axis=1))
    d_ = build_multiindex(d_)
    e1x = ["homicide_rate_log_lag1"] + CONTROLS
    w1 = d_[["idx"] + e1x].dropna()
    r1 = fe_wild_test(w1, "idx", "homicide_rate_log_lag1", e1x, B=B_BOOT)
    e2x = ["idx"] + CONTROLS
    w2 = d_[["fdi_percent_gdp"] + e2x].dropna()
    r2 = fe_wild_test(w2, "fdi_percent_gdp", "idx", e2x, B=B_BOOT)
    by_content[name] = {"dims": dims, "EQ1": r1, "EQ2": r2}
    print(f"    {name:<26} EQ1 beta {r1['coef']:+.3f} (p clustered {r1['p_cl']:.3f}, wild {r1['p_wild']:.3f}) | "
          f"EQ2 beta {r2['coef']:+.3f} (p clustered {r2['p_cl']:.3f}, wild {r2['p_wild']:.3f})")
p_vr, p_nc = by_content["violence_related"]["EQ1"]["p_wild"], by_content["no_crime_content"]["EQ1"]["p_wild"]

# ════════════════════════════════════════════════════════════════════════
# 5. SENSITIVITY TO THE NUMBER OF SOURCES
# ════════════════════════════════════════════════════════════════════════
section("14-F — SENSITIVITY TO COUNTRY-YEARS WITH FEW SOURCES")
src = ci.pivot_table(index=[ENTITY_COL, TIME_COL], columns="dim", values="n_sources").reset_index()
src["min_sources"] = src[DIMS].min(axis=1)
base = panel.merge(src[[ENTITY_COL, TIME_COL, "min_sources"]], on=[ENTITY_COL, TIME_COL])
by_sources = {}
for k in (1, 3, 5):
    d_ = build_multiindex(base[base["min_sources"] >= k].copy())
    e1x = ["homicide_rate_log_lag1"] + CONTROLS
    w1 = d_[["inst_avg"] + e1x].dropna()
    e2x = ["inst_avg"] + CONTROLS
    w2 = d_[["fdi_percent_gdp"] + e2x].dropna()
    if len(w1) < 100 or len(w2) < 100:
        continue
    r1 = fe_wild_test(w1, "inst_avg", "homicide_rate_log_lag1", e1x, B=B_BOOT)
    r2 = fe_wild_test(w2, "fdi_percent_gdp", "inst_avg", e2x, B=B_BOOT)
    by_sources[f"min_{k}"] = {"EQ1": r1, "EQ2": r2, "n_country_years": int(len(d_))}
    print(f"    every dimension has >= {k} sources (n={len(d_)}): EQ1 beta {r1['coef']:+.3f} (wild p {r1['p_wild']:.3f}) | "
          f"EQ2 beta {r2['coef']:+.3f} (wild p {r2['p_wild']:.3f})")

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("14-G — FIGURE")
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
short = {"rule_of_law": "RL", "political_stability": "PS", "control_corruption": "CC",
         "voice_accountability": "VA", "government_effectiveness": "GE", "regulatory_quality": "RQ"}
xs = np.arange(len(DIMS))
ax[0, 0].bar(xs - 0.2, [noise_tab[d]["mean_se"] for d in DIMS], 0.4, color=AMBER, label="mean measurement SE")
ax[0, 0].bar(xs + 0.2, [noise_tab[d]["sd_within_country"] for d in DIMS], 0.4, color=BLUE, label="within-country SD of the score")
ax[0, 0].set_xticks(xs); ax[0, 0].set_xticklabels([short[d] for d in DIMS])
ax[0, 0].set_ylabel("points on the 0-100 scale"); ax[0, 0].legend(fontsize=8)
ax[0, 0].set_title("(a) Measurement noise is as large as the variation FE models use", fontsize=10)
w_ = 0.27
for j, lag in enumerate(("1", "5", "10")):
    ax[0, 1].bar(xs + (j - 1) * w_, [change_share[d][lag]["share_significant"] * 100 for d in DIMS], w_, label=f"{lag} yr", color=[GREY, BLUE, GREEN][j])
ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels([short[d] for d in DIMS])
ax[0, 1].set_ylabel("% of changes outside the 90% margin"); ax[0, 1].legend(title="change over", fontsize=8)
ax[0, 1].set_title("(b) Most observed changes cannot be told apart from noise", fontsize=10)
for d, col in (("voice_accountability", RED_C), ("rule_of_law", BLUE), ("control_corruption", GREY)):
    s = ci[(ci["dim"] == d) & (ci[ENTITY_COL] == "SLV")].sort_values(TIME_COL)
    ax[1, 0].plot(s[TIME_COL], s["score"], color=col, lw=1.8, label=short[d])
    ax[1, 0].fill_between(s[TIME_COL], s["lb"], s["ub"], color=col, alpha=0.15)
ax[1, 0].axvline(2020.5, color="black", ls=":", lw=1)
ax[1, 0].set_ylabel("score (shaded: 90% interval)"); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("(c) El Salvador: only the Voice & Accountability fall clears its margin", fontsize=10)
names = list(GROUPS)
coefs = [by_content[n]["EQ1"]["coef"] for n in names]
errs = [1.96 * by_content[n]["EQ1"]["se_cl"] for n in names]
colors = [RED_C if n == "violence_related" else GREY for n in names]
ax[1, 1].barh(range(len(names)), coefs, xerr=errs, color=colors, capsize=3)
ax[1, 1].set_yticks(range(len(names)))
ax[1, 1].set_yticklabels([f"{n.replace('_', ' ')}\n(wild p={by_content[n]['EQ1']['p_wild']:.3f})" for n in names], fontsize=8)
ax[1, 1].axvline(0, color="black", lw=0.8); ax[1, 1].invert_yaxis()
ax[1, 1].set_xlabel("EQ1 coefficient of lagged log homicides (z-scored index, +/-1.96 clustered SE)")
ax[1, 1].set_title("(d) EQ1 by dimension group (post-hoc grouping, exploratory)", fontsize=10)
fig.tight_layout()
fig.savefig(DIRS["figures"] / "19_wgi_measurement.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/19_wgi_measurement.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
save_json({
    "n_obs_ci": int(len(ci)),
    "assumption": "errors treated as independent across years and dimensions; reliabilities are lower bounds",
    "noise_by_dimension": noise_tab,
    "change_distinguishable_share": change_share,
    "changes_2020_2024": chg.to_dict(orient="records"),
    "el_salvador_2020_2024": slv.to_dict(orient="records"),
    "reliability_worst_case": reliability,
    "eq1_eq2_by_content": by_content,
    "sensitivity_min_sources": by_sources,
    "summary": {
        "n_changes_2020_2024": int(len(chg)), "n_distinguishable_2020_2024": n_dist,
        "slv_distinguishable_dims": [r["dim"] for _, r in slv.iterrows() if r["distinguishable"]],
        "mean_noise_to_within_sd": float(np.mean([noise_tab[d]["noise_to_within_sd"] for d in DIMS])),
        "index_worst_case_reliability": lam_idx,
        "eq1_p_wild_violence_related": p_vr, "eq1_p_wild_no_crime_content": p_nc,
        "eq1_p_wild_government_effectiveness": by_content["government_effectiveness"]["EQ1"]["p_wild"],
        "grouping_is_post_hoc": True,
        "n_bootstrap": B_BOOT,
    },
}, DIRS["json"] / "14_wgi_measurement.json")
print(bold(f"\nSummary: noise/within SD {np.mean([noise_tab[d]['noise_to_within_sd'] for d in DIMS]):.2f} | worst-case index reliability {lam_idx:.2f} | "
           f"EQ1 wild p: violence-related {p_vr:.3f} vs no-crime-content {p_nc:.3f}"))
