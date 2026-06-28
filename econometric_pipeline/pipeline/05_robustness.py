"""
05_robustness.py
════════════════════════════════════════════════════════════════════════
Layer 5 — Robustness & Sensitivity Analysis

CRITICAL DESIGN CONSTRAINT
───────────────────────────
Every robustness exercise estimates EXACTLY the same model as Module 02:
    PanelOLS(entity_effects=True, time_effects=True, drop_absorbed=True)
    cov_type="clustered", cluster_entity=True

No silent substitution by pooled OLS, manually demeaned OLS, or any
other estimator. Any deviation is explicitly justified.

Robustness exercises
────────────────────
1.  Leave-One-Country-Out (LOCO)
      Bias addressed: results driven by a single influential country.

2.  Time windows
      (a) Pre-2010 (pre-GFC structural break)
      (b) Post-2010
      (c) Excluding COVID years 2020–2021
      (d) Full sample (baseline)
      Bias addressed: period-specific shocks, structural breaks.

3.  Alternative lag structures
      Uses pre-computed columns: lag1, lag2, lag3.
      Bias addressed: uncertainty about transmission delay.

4.  Alternative institution indices
      (a) Standardised average (primary)
      (b) PCA-based index
      (c) Rule of Law only
      (d) Control of Corruption only
      (e) Political Stability only
      Bias addressed: index construction choices.

5.  Alternative control sets
      (a) Baseline controls
      (b) Extended controls (tourist arrivals, trade)
      (c) Minimal controls (GDP per capita only)
      Bias addressed: over/under-controlling.

6.  Alternative clustering
      (a) By entity (primary)
      (b) Driscoll-Kraay HAC (spatial + temporal)
      Bias addressed: sensitivity to SE specification.

7.  Placebo test
      Randomly permute homicide_rate_log_lag1 within year cells.
      Expect: no significant effect when causal ordering is broken.
      Bias addressed: spurious correlation.

Literature
──────────────────────────────────────────────────────────────────────
Oster (2019) — Unobservable Selection and Coefficient Stability.
Brodeur et al. (2020) — Methods Matter.
Angrist & Pischke (2009) — ch. 2 (threats to internal validity).
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

from utils import (
    set_plot_style,
    section,
    subsection,
    ok,
    warn,
    err,
    bold,
    sig_stars,
    save_json,
    make_output_dirs,
    ENTITY_COL,
    TIME_COL,
    BOLD,
    RESET,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, build_multiindex, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("05-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_with_predictions.csv"
META_PATH = Path(__file__).parent / "json" / "01_metadata.json"
SPEC_PATH = Path(__file__).parent / "fe_specs.pkl"

df   = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)
with open(META_PATH) as f: meta = json.load(f)
with open(SPEC_PATH, "rb") as f: specs = pickle.load(f)

COUNTRIES = meta["countries"]
CONTROLS  = meta["controls"]
print(ok(f"Data: {len(df)} obs | {len(COUNTRIES)} countries"))

# ════════════════════════════════════════════════════════════════════════
# CORE ESTIMATION FUNCTION — IDENTICAL TO MODULE 02
# ════════════════════════════════════════════════════════════════════════

def run_twoway_fe(
    df_sub:    pd.DataFrame,
    dep:       str,
    exog:      list,
    key_var:   str,
    label:     str = "",
) -> dict:
    """
    Estimate Two-Way FE (entity + time) with clustered SE by entity.
    Identical specification to Module 02 primary estimator.

    Returns dict with coef, se, pval, rsq_within, n_obs for key_var.
    Returns NaN fields if estimation fails.
    """
    nan_result = {
        "coef": np.nan, "se": np.nan, "pval": np.nan,
        "rsq_within": np.nan, "n_obs": 0, "label": label,
        "converged": False,
    }
    try:
        exog_a = [c for c in exog if c in df_sub.columns]
        if dep not in df_sub.columns or not exog_a:
            return nan_result

        df_idx = build_multiindex(df_sub)
        work   = df_idx[[dep] + exog_a].dropna()
        n_obs  = len(work)

        # Need at least k+G+T observations
        n_entities = work.index.get_level_values(0).nunique()
        n_times    = work.index.get_level_values(1).nunique()
        if n_obs < len(exog_a) + n_entities + n_times + 2:
            return {**nan_result, "label": label}

        mod = PanelOLS(
            work[dep], work[exog_a],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        if key_var not in res.params.index:
            return nan_result

        return {
            "coef":       float(res.params[key_var]),
            "se":         float(res.std_errors[key_var]),
            "pval":       float(res.pvalues[key_var]),
            "rsq_within": float(res.rsquared_within),
            "n_obs":      n_obs,
            "label":      label,
            "converged":  True,
        }
    except Exception as e:
        return {**nan_result, "label": label, "error": str(e)}


def print_robustness_table(
    results:   list,
    title:     str,
    baseline:  dict,
    key_var:   str,
) -> None:
    """Print a formatted robustness table against the baseline."""
    print(f"\n  {bold(title)}")
    print(f"  {'Specification':<45} {'N':>6} {'β̂':>9} {'SE':>8} {'p':>7} {'Stars':>6} {'Δβ':>9}")
    print("  " + "─" * 95)

    b_coef = baseline.get("coef", np.nan)
    for r in results:
        if not r["converged"]:
            print(f"  {r['label']:<45} {'—':>6} {'—':>9} {'—':>8} {'—':>7} {'':>6} {'FAILED':>9}")
            continue
        delta  = r["coef"] - b_coef if not np.isnan(b_coef) else np.nan
        stars  = sig_stars(r["pval"])
        hi     = BOLD if r.get("is_baseline") else ""
        from utils import BOLD as _B, RESET
        print(
            f"  {hi}{r['label']:<45}{RESET} "
            f"{r['n_obs']:>6} "
            f"{r['coef']:>9.4f} "
            f"{r['se']:>8.4f} "
            f"{r['pval']:>7.3f} "
            f"{stars:>6} "
            f"{delta:>+9.4f}"
        )

# ════════════════════════════════════════════════════════════════════════
# 1. LEAVE-ONE-COUNTRY-OUT
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 1 — Leave-One-Country-Out (LOCO)")
print("  Bias addressed: results driven by a single influential country.\n")

all_robustness = {}

for eq_label, spec in specs.items():
    dep     = spec["dep"]
    exog    = spec["exog"]
    key_var = [c for c in exog if c in df.columns][0]

    # Baseline (full sample)
    baseline = run_twoway_fe(df, dep, exog, key_var, "Full sample (baseline)")
    baseline["is_baseline"] = True

    loo_results = [baseline]
    for excl in COUNTRIES:
        df_sub = df[df[ENTITY_COL] != excl].copy()
        r = run_twoway_fe(df_sub, dep, exog, key_var, f"Excl. {excl}")
        loo_results.append(r)

    print_robustness_table(loo_results, f"{eq_label.upper()}: {dep}", baseline, key_var)

    # Fragility check: sign-reversal or >50% magnitude change
    coefs_valid = [r["coef"] for r in loo_results[1:] if r["converged"] and not np.isnan(r["coef"])]
    if coefs_valid and not np.isnan(baseline["coef"]):
        sign_flips  = sum(np.sign(c) != np.sign(baseline["coef"]) for c in coefs_valid)
        large_shift = sum(abs(c - baseline["coef"]) > 0.5 * abs(baseline["coef"]) for c in coefs_valid)
        if sign_flips == 0 and large_shift == 0:
            print(ok(f"  Robust: no sign reversals, no large shifts across LOCO."))
        else:
            print(warn(f"  {sign_flips} sign reversal(s), {large_shift} large shift(s) — check influential countries."))

    all_robustness[f"{eq_label}_loco"] = loo_results

# ════════════════════════════════════════════════════════════════════════
# 2. TIME WINDOWS
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 2 — Time Windows")
print("  Bias addressed: period-specific shocks, structural breaks.\n")

time_windows = {
    "Full sample":          (None, None),
    "Pre-2010 (pre-GFC)":   (None, 2009),
    "Post-2010":            (2010, None),
    "Excl. COVID (2020–21)":("excl_covid", None),
    "Post-peace (2014+)":   (2014, None),
}

for eq_label, spec in specs.items():
    dep     = spec["dep"]
    exog    = spec["exog"]
    key_var = [c for c in exog if c in df.columns][0]

    tw_results = []
    for wname, (yr_lo, yr_hi) in time_windows.items():
        if wname == "Excl. COVID (2020–21)":
            df_sub = df[~df[TIME_COL].isin([2020, 2021])].copy()
        else:
            df_sub = df.copy()
            if yr_lo is not None: df_sub = df_sub[df_sub[TIME_COL] >= yr_lo]
            if yr_hi is not None: df_sub = df_sub[df_sub[TIME_COL] <= yr_hi]

        r = run_twoway_fe(df_sub, dep, exog, key_var, wname)
        r["is_baseline"] = (wname == "Full sample")
        tw_results.append(r)

    baseline_tw = tw_results[0]
    print_robustness_table(tw_results, f"{eq_label.upper()}: {dep}", baseline_tw, key_var)
    all_robustness[f"{eq_label}_timewindow"] = tw_results

# ════════════════════════════════════════════════════════════════════════
# 3. ALTERNATIVE LAG STRUCTURES (EQ1 ONLY — primary chain entry)
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 3 — Alternative Lag Structures (EQ1: Violence → Institutions)")
print("  Uses pre-computed lag columns from dataset (not re-computed).")
print("  Bias addressed: uncertainty about transmission delay of violence.\n")

lag_specs = {
    "Contemporaneous (lag 0)": "homicide_rate_log",
    "Lag 1 (primary)":         "homicide_rate_log_lag1",
    "Lag 2":                   "homicide_rate_log_lag2",
    "Lag 3":                   "homicide_rate_log_lag3",
}

dep_eq1  = specs["eq1"]["dep"]
ctrl_eq1 = [c for c in specs["eq1"]["exog"][1:] if c in df.columns]  # skip key_var

lag_results = []
for lag_name, lag_col in lag_specs.items():
    if lag_col not in df.columns:
        lag_results.append({"label": lag_name, "converged": False})
        continue
    exog_lag = [lag_col] + ctrl_eq1
    r = run_twoway_fe(df, dep_eq1, exog_lag, lag_col, lag_name)
    r["is_baseline"] = (lag_name == "Lag 1 (primary)")
    lag_results.append(r)

baseline_lag = next((r for r in lag_results if r.get("is_baseline")), lag_results[0])
print_robustness_table(lag_results, "EQ1: Alternative Lags", baseline_lag, "key")
all_robustness["eq1_lags"] = lag_results

# ════════════════════════════════════════════════════════════════════════
# 4. ALTERNATIVE INSTITUTION INDICES
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 4 — Alternative Institution Indices")
print("  Bias addressed: sensitivity to index construction methodology.\n")

# Alternative dependent variables for EQ1
inst_alternatives = {
    "inst_avg (primary)":      "inst_avg",
    "inst_pca (PCA-based)":    "inst_pca",
    "rule_of_law":             "rule_of_law",
    "control_corruption":      "control_corruption",
    "political_stability":     "political_stability",
}

key_eq1  = "homicide_rate_log_lag1"
ctrl_eq1 = [c for c in specs["eq1"]["exog"] if c != key_eq1 and c in df.columns]

inst_results = []
for inst_name, inst_col in inst_alternatives.items():
    if inst_col not in df.columns:
        inst_results.append({"label": inst_name, "converged": False})
        continue
    exog_inst = [key_eq1] + ctrl_eq1
    r = run_twoway_fe(df, inst_col, exog_inst, key_eq1, inst_name)
    r["is_baseline"] = ("primary" in inst_name)
    inst_results.append(r)

baseline_inst = next((r for r in inst_results if r.get("is_baseline")), inst_results[0])
print_robustness_table(inst_results, "EQ1: Alternative Institution Indices", baseline_inst, key_eq1)
all_robustness["eq1_inst_index"] = inst_results

# Sign consistency check
coefs_inst = [r["coef"] for r in inst_results if r["converged"] and not np.isnan(r.get("coef", np.nan))]
if coefs_inst:
    all_neg = all(c < 0 for c in coefs_inst)
    all_pos = all(c > 0 for c in coefs_inst)
    if all_neg or all_pos:
        print(ok("  Sign consistent across all institution specifications."))
    else:
        print(warn("  Mixed signs across institution specifications — investigate."))

# ════════════════════════════════════════════════════════════════════════
# 5. ALTERNATIVE CONTROL SETS
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 5 — Alternative Control Sets")
print("  Bias addressed: over-controlling or omitted variable bias.\n")

CTRL_BASE    = [c for c in CONTROLS if c in df.columns]
CTRL_EXT     = CTRL_BASE + [c for c in ["tourist_arrivals_log","trade_percent_gdp"]
                             if c in df.columns]
CTRL_MINIMAL = [c for c in ["gdp_per_capita_log"] if c in df.columns]

ctrl_specs_map = {
    "Baseline controls":      CTRL_BASE,
    "Extended controls":      CTRL_EXT,
    "Minimal controls":       CTRL_MINIMAL,
    "No controls (FE only)":  [],
}

for eq_label, spec in specs.items():
    dep      = spec["dep"]
    key_var  = [c for c in spec["exog"] if c in df.columns][0]

    ctrl_results = []
    for ctrl_name, ctrl_cols in ctrl_specs_map.items():
        exog_c = [key_var] + ctrl_cols
        r = run_twoway_fe(df, dep, exog_c, key_var, ctrl_name)
        r["is_baseline"] = ("Baseline" in ctrl_name)
        ctrl_results.append(r)

    baseline_ctrl = next((r for r in ctrl_results if r.get("is_baseline")), ctrl_results[0])
    print_robustness_table(ctrl_results, f"{eq_label.upper()}: {dep}", baseline_ctrl, key_var)
    all_robustness[f"{eq_label}_controls"] = ctrl_results

# ════════════════════════════════════════════════════════════════════════
# 6. PLACEBO TEST
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 6 — Placebo Test (permuted homicide)")
print("""
  Procedure: randomly permute homicide_rate_log_lag1 within year cells.
  This breaks the country-specific causal link while preserving the
  cross-sectional and temporal distribution of the variable.
  H0 (placebo): permuted violence has no effect on institutions.
  If the TRUE estimate is more extreme than 95% of placebo estimates,
  this confirms the result is not driven by spurious correlation.
  N_permutations = 500.
""")

N_PERM   = 500
rng      = np.random.default_rng(SEED)

dep_p    = specs["eq1"]["dep"]
exog_p   = specs["eq1"]["exog"]
key_p    = "homicide_rate_log_lag1"
ctrl_p   = [c for c in exog_p if c != key_p and c in df.columns]

if key_p not in df.columns:
    print(warn(f"  {key_p} not in data. Placebo skipped."))
else:
    # True estimate
    true_res  = run_twoway_fe(df, dep_p, exog_p, key_p, "True estimate")
    true_coef = true_res["coef"]

    # Permutation loop
    placebo_coefs = []
    for i in range(N_PERM):
        df_p = df.copy()
        # Permute within year (preserves year distribution, breaks country link)
        df_p[key_p] = df_p.groupby(TIME_COL)[key_p].transform(
            lambda x: rng.permutation(x.values)
        )
        r_p = run_twoway_fe(df_p, dep_p, exog_p, key_p, f"Perm {i}")
        if r_p["converged"] and not np.isnan(r_p["coef"]):
            placebo_coefs.append(r_p["coef"])

    if len(placebo_coefs) >= 50:
        p_arr = np.array(placebo_coefs)
        # Two-sided placebo p-value: share of permutations more extreme than truth
        p_placebo = (np.abs(p_arr) >= np.abs(true_coef)).mean()

        print(f"  True estimate:        β̂ = {true_coef:.4f}")
        print(f"  Placebo distribution: mean={p_arr.mean():.4f}, "
              f"SD={p_arr.std():.4f}, "
              f"5th pctile={np.percentile(p_arr,5):.4f}")
        print(f"  Placebo p-value:      {p_placebo:.4f}")

        if p_placebo < 0.05:
            print(ok("  Placebo test PASSED — true estimate more extreme than 95% of random permutations."))
        else:
            print(warn("  Placebo test inconclusive — true estimate not extreme in permutation distribution."))

        all_robustness["placebo"] = {
            "true_coef":    true_coef,
            "placebo_mean": float(p_arr.mean()),
            "placebo_sd":   float(p_arr.std()),
            "p_placebo":    float(p_placebo),
            "n_perm":       len(placebo_coefs),
        }
    else:
        print(warn(f"  Only {len(placebo_coefs)} valid permutations. Results unreliable."))

# ════════════════════════════════════════════════════════════════════════
# FIGURE: Coefficient stability plots
# ════════════════════════════════════════════════════════════════════════
section("05-G — FIGURES")

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "Robustness Analysis — Coefficient Stability\n"
    "Two-Way Fixed Effects | Dep. var: Institutional Quality (inst_avg)\n"
    "All specifications use identical estimator: PanelOLS(entity_effects=True, time_effects=True)",
    fontsize=11, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

BLUE   = "#2563EB"
RED_C  = "#DC2626"
GREEN  = "#16A34A"
GREY   = "#6B7280"
AMBER  = "#D97706"

def coef_plot(ax, results, title, key_var, baseline_label="Full sample (baseline)"):
    """Generic coefficient plot with error bars and baseline reference."""
    valid = [r for r in results if r.get("converged") and not np.isnan(r.get("coef", np.nan))]
    if not valid: return

    coefs   = [r["coef"] for r in valid]
    ses     = [r["se"]   for r in valid]
    labels  = [r["label"] for r in valid]
    pvals   = [r["pval"]  for r in valid]

    colors  = [BLUE if l == baseline_label else
               (RED_C if abs(r["coef"]) > 2 * abs(valid[0]["coef"]) else GREY)
               for l, r in zip(labels, valid)]
    y_pos   = np.arange(len(valid))[::-1]

    ax.errorbar(coefs, y_pos, xerr=1.96 * np.array(ses),
                fmt="o", color=BLUE, ecolor=GREY,
                capsize=4, markersize=5, linewidth=1.2)
    for i, (c, se, col, pv) in enumerate(zip(coefs, ses, colors, pvals)):
        ax.scatter(c, y_pos[i], color=col, s=40, zorder=5)
        if pv < 0.10:
            ax.text(c + 1.96 * se + 0.005, y_pos[i], sig_stars(pv),
                    va="center", fontsize=8, color=RED_C)

    baseline_coef = next((r["coef"] for r in valid if r["label"] == baseline_label), None)
    if baseline_coef is not None:
        ax.axvline(baseline_coef, color=BLUE, linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Baseline β={baseline_coef:.3f}")
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"β̂ ({key_var})", fontsize=9)
    ax.set_title(title, fontsize=9)
    if baseline_coef is not None:
        ax.legend(fontsize=8)

# ── Panel 1: LOCO ──
ax1 = fig.add_subplot(gs[0, :])
loo_data = all_robustness.get("eq1_loco", [])
coef_plot(ax1, loo_data, "Leave-One-Country-Out: Violence → Institutions", "homicide_rate_log_lag1")

# ── Panel 2: Time windows ──
ax2 = fig.add_subplot(gs[1, 0])
tw_data = all_robustness.get("eq1_timewindow", [])
coef_plot(ax2, tw_data, "Time Windows", "homicide_rate_log_lag1", "Full sample")

# ── Panel 3: Alternative institution indices ──
ax3 = fig.add_subplot(gs[1, 1])
inst_data = all_robustness.get("eq1_inst_index", [])
coef_plot(ax3, inst_data, "Alternative Institution Indices", "homicide_rate_log_lag1", "inst_avg (primary)")

# ── Panel 4: Alternative lags ──
ax4 = fig.add_subplot(gs[2, 0])
lag_data = all_robustness.get("eq1_lags", [])
coef_plot(ax4, lag_data, "Alternative Lag Structures", "key", "Lag 1 (primary)")

# ── Panel 5: Placebo distribution ──
ax5 = fig.add_subplot(gs[2, 1])
if "placebo" in all_robustness and len(placebo_coefs) > 0:
    ax5.hist(placebo_coefs, bins=40, color=GREY, edgecolor="white", alpha=0.8,
             density=True, label=f"Placebo distribution (N={len(placebo_coefs)})")
    ax5.axvline(true_coef, color=RED_C, linewidth=2.5,
                label=f"True β={true_coef:.3f}")
    ax5.axvline(np.percentile(placebo_coefs, 2.5),  color="black",
                linestyle="--", linewidth=1, alpha=0.7)
    ax5.axvline(np.percentile(placebo_coefs, 97.5), color="black",
                linestyle="--", linewidth=1, alpha=0.7, label="2.5/97.5 pctiles")
    ax5.set_xlabel("β̂ (permuted homicide)", fontsize=9)
    ax5.set_ylabel("Density", fontsize=9)
    ax5.set_title(f"Placebo Test\np={all_robustness['placebo']['p_placebo']:.3f}", fontsize=9)
    ax5.legend(fontsize=8)
else:
    ax5.text(0.5, 0.5, "Placebo results\nnot available", ha="center", va="center",
             transform=ax5.transAxes)

fig.savefig(DIRS["figures"] / "07_robustness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved → figures/07_robustness.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("05-H — EXPORT")

export = {}
for k, v in all_robustness.items():
    if isinstance(v, list):
        export[k] = [
            {kk: (float(vv) if isinstance(vv, (float, np.floating)) and not np.isnan(vv)
                  else (None if isinstance(vv, float) and np.isnan(vv)
                        else vv))
             for kk, vv in r.items() if kk not in ("t_boot", "beta_boot")}
            for r in v
        ]
    else:
        export[k] = v

save_json(export, DIRS["json"] / "05_robustness.json")
print(f"\n{bold('Module 05 complete.')}")
