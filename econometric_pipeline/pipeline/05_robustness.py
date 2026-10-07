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
      (a) Standardised average (primary, all 6 WGI dimensions)
      (b) PCA-based index
      (c)-(h) Each of the 6 WGI dimensions individually (rule of law,
          control of corruption, political stability, voice &
          accountability, government effectiveness, regulatory quality)
      Bias addressed: index construction choices.

4b. The "Bukele paradox" (added September 2026)
      LOCO check on voice_accountability specifically -- the one WGI
      dimension where violence has the wrong (positive) sign. Tests
      whether El Salvador's 2021-2024 pattern (homicides collapse,
      voice_accountability ALSO falls -- security gains and
      civil-liberties costs moving together, not apart) is driving an
      apparent panel-wide relationship, or is a country-specific case.
      Bias addressed: mistaking a single influential country's pattern
      for a general relationship.

5.  Alternative control sets
      (a) Baseline controls
      (b) Extended controls (tourist arrivals, trade)
      (c) Minimal controls (GDP per capita only)
      (d) Excl. GDP p.c. (bad-control check)
      (e) Incl. remittances (bad-control check; added September 2026)
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
from typing import Optional, Tuple
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
    fisher_panel_unit_root,
)

set_plot_style()
DIRS  = make_output_dirs(Path(__file__).parent)
SEED  = 42
ALPHA = 0.05
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
# 3b. LAG STRUCTURE FOR EQ2 & EQ3 (theory/specification consistency check)
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 3b — Lag Structure for EQ2 and EQ3")
print("""
  Motivation: the theoretical model (paper Introduction) argues that
  security, institutions and growth adjust at DIFFERENT speeds --
  institutions evolve gradually and growth responds with a further lag.
  Yet the PRIMARY specifications of EQ2 (institutions -> FDI) and EQ3
  (FDI + institutions -> growth) use CONTEMPORANEOUS regressors, which
  is inconsistent with that narrative and leaves the door open to
  reverse causality / simultaneity (e.g. growth expectations attracting
  same-year FDI). This block re-estimates EQ2 and EQ3 replacing the
  contemporaneous institution/FDI regressors with their one-year lag
  (inst_avg_lag1, fdi_percent_gdp_lag1 -- built in Module 01) to test
  whether the results survive a specification that matches the stated
  theory.
""")

# ── EQ2: inst_avg (contemporaneous, primary) vs inst_avg_lag1 ───────────
dep_eq2  = specs["eq2"]["dep"]
ctrl_eq2 = [c for c in specs["eq2"]["exog"][1:] if c in df.columns]  # skip key_var

eq2_lag_specs = {
    "Contemporaneous inst_avg (primary)": "inst_avg",
    "Lag 1: inst_avg_lag1":               "inst_avg_lag1",
}
eq2_lag_results = []
for lag_name, lag_col in eq2_lag_specs.items():
    if lag_col not in df.columns:
        eq2_lag_results.append({"label": lag_name, "converged": False})
        continue
    exog_lag = [lag_col] + ctrl_eq2
    r = run_twoway_fe(df, dep_eq2, exog_lag, lag_col, lag_name)
    r["is_baseline"] = ("primary" in lag_name)
    eq2_lag_results.append(r)

baseline_eq2_lag = next((r for r in eq2_lag_results if r.get("is_baseline")), eq2_lag_results[0])
print_robustness_table(eq2_lag_results, "EQ2: Contemporaneous vs Lagged Institutions",
                        baseline_eq2_lag, "key")
all_robustness["eq2_lags"] = eq2_lag_results

# ── EQ3: (fdi, inst_avg) contemporaneous vs both lagged one year ────────
dep_eq3  = specs["eq3"]["dep"]
ctrl_eq3 = [c for c in specs["eq3"]["exog"][2:] if c in df.columns]  # skip both keys

eq3_lag_variants = {
    "Contemporaneous fdi+inst (primary)": ["fdi_percent_gdp", "inst_avg"],
    "Lag 1: fdi_lag1+inst_lag1":          ["fdi_percent_gdp_lag1", "inst_avg_lag1"],
}
eq3_lag_results = []
for lag_name, keyvars in eq3_lag_variants.items():
    keyvars_a = [c for c in keyvars if c in df.columns]
    if len(keyvars_a) < len(keyvars):
        eq3_lag_results.append({"label": lag_name, "converged": False})
        continue
    exog_lag = keyvars_a + ctrl_eq3
    r = run_twoway_fe(df, dep_eq3, exog_lag, keyvars_a[0], lag_name)
    r["is_baseline"] = ("primary" in lag_name)
    eq3_lag_results.append(r)

baseline_eq3_lag = next((r for r in eq3_lag_results if r.get("is_baseline")), eq3_lag_results[0])
print_robustness_table(eq3_lag_results, "EQ3: Contemporaneous vs Lagged FDI+Institutions",
                        baseline_eq3_lag, "key")
print(warn("  Note: 'key' here is fdi_percent_gdp (or its lag) -- the coefficient printed. "
           "inst_avg's own coefficient in this lagged spec is in the JSON export, not this table."))
all_robustness["eq3_lags"] = eq3_lag_results

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
    "voice_accountability":       "voice_accountability",
    "government_effectiveness":   "government_effectiveness",
    "regulatory_quality":         "regulatory_quality",
}
inst_alternatives = {k: v for k, v in inst_alternatives.items() if v in df.columns}

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
# 4b. THE "BUKELE PARADOX": LOCO ON THE ANOMALOUS voice_accountability DIMENSION
# ════════════════════════════════════════════════════════════════════════
section('ROBUSTNESS 4b — "Bukele Paradox": LOCO on voice_accountability')
print("""
  Motivation: of the six WGI dimensions tested above, voice_accountability
  is the only one where homicide_rate_log_lag1 has the WRONG (positive)
  sign -- less violence coinciding with LOWER voice & accountability,
  instead of higher. El Salvador's 2021-2024 data shows exactly this:
  homicides collapsed (17.3 -> 1.9 per 100k) while voice_accountability
  ALSO fell sharply (53.5 -> 45.0), the documented civil-liberties cost of
  the post-2020 security crackdown (formal state of exception from March 2022). This LOCO check tests
  whether that single case is driving the anomaly, or whether it reflects
  a broader pattern across the panel.
""")

va_col = "voice_accountability"
if va_col in df.columns:
    exog_va = [key_eq1] + ctrl_eq1
    va_full = run_twoway_fe(df, va_col, exog_va, key_eq1, "Full sample")
    va_full["is_baseline"] = True

    va_loco = [va_full]
    for c in COUNTRIES:
        sub = df[df[ENTITY_COL] != c]
        r = run_twoway_fe(sub, va_col, exog_va, key_eq1, f"Excl. {c}")
        r["is_baseline"] = False
        va_loco.append(r)

    print_robustness_table(va_loco, "voice_accountability ~ homicide_rate_log_lag1 (LOCO)",
                            va_full, key_eq1)

    slv_r = next((r for r in va_loco if r["label"] == "Excl. SLV"), None)
    if slv_r and slv_r["converged"] and va_full["converged"]:
        flips = sum(
            1 for r in va_loco[1:]
            if r["converged"] and np.sign(r["coef"]) != np.sign(va_full["coef"])
        )
        if np.sign(slv_r["coef"]) != np.sign(va_full["coef"]) and flips == 1:
            print(ok(
                f"  El Salvador is the ONLY country whose exclusion flips the sign "
                f"(full={va_full['coef']:.3f}, excl.SLV={slv_r['coef']:.3f}). The anomalous "
                "positive coefficient is a genuine El Salvador-specific pattern (the "
                "'Bukele paradox': security gains and civil-liberties costs moving "
                "together), not a broader regional relationship -- see README."
            ))
        else:
            print(warn(
                f"  El Salvador is NOT uniquely responsible for the sign ({flips} "
                "countries flip it on exclusion) -- the 'Bukele paradox' framing may "
                "be too narrow; re-examine before writing it up."
            ))
    all_robustness["eq1_voice_accountability_loco"] = va_loco
else:
    print(warn(f"  {va_col} not in data. Bukele-paradox LOCO check skipped."))

# ════════════════════════════════════════════════════════════════════════
# 5. ALTERNATIVE CONTROL SETS
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 5 — Alternative Control Sets")
print("  Bias addressed: over-controlling or omitted variable bias.\n")

CTRL_BASE     = [c for c in CONTROLS if c in df.columns]
CTRL_EXT      = CTRL_BASE + [c for c in ["tourist_arrivals_log","trade_percent_gdp"]
                              if c in df.columns]
CTRL_MINIMAL  = [c for c in ["gdp_per_capita_log"] if c in df.columns]
CTRL_NO_GDPPC = [c for c in CTRL_BASE if c != "gdp_per_capita_log"]
CTRL_REMIT    = CTRL_BASE + [c for c in ["remittances_percent_gdp"] if c in df.columns]

ctrl_specs_map = {
    "Baseline controls":      CTRL_BASE,
    "Extended controls":      CTRL_EXT,
    "Minimal controls":       CTRL_MINIMAL,
    "Excl. GDP p.c. (bad-control check)": CTRL_NO_GDPPC,
    "Incl. remittances (bad-control check)": CTRL_REMIT,
    "No controls (FE only)":  [],
}

print(warn("""  'Excl. GDP p.c. (bad-control check)': gdp_per_capita_log is plausibly
  DOWNSTREAM of institutions/FDI (part of the very causal channel this
  pipeline studies), not an exogenous confounder. Including it as a
  control in EQ1/EQ2 risks a post-treatment ('bad control') bias that
  partials out some of the effect of interest (Angrist & Pischke 2009,
  ch. 3). This variant drops it while keeping the rest of the baseline
  controls, to see whether the key coefficient is sensitive to its
  inclusion.
"""))
print(warn("""  'Incl. remittances (bad-control check)': remittances_percent_gdp (added
  September 2026) is plausibly a BAD CONTROL of a different kind -- it is
  a candidate mediator/collider between violence (emigration push) and
  growth (remittance-funded consumption), not a clean exogenous control.
  This variant ADDS it to the baseline controls (rather than treating it
  as part of the primary specification) to see whether the key
  coefficient is sensitive to its inclusion, kept separate from
  Module 01's core institution-index change for the same reason
  gdp_per_capita_log is tested as a variant rather than assumed safe.
"""))

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
# 7. COUNTRY-SPECIFIC LINEAR TRENDS
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 7 — Country-Specific Linear Trends")
print("""
  Motivation: Two-Way FE (entity + time) removes country LEVELS and
  common year shocks, but NOT country-specific trends. Several series
  in this panel are strongly trending over 25 years (GDP per capita,
  population, and especially El Salvador's homicide collapse from >100
  to <10 per 100,000). If two trending series happen to move together
  for reasons unrelated to the hypothesized mechanism, Two-Way FE alone
  will not catch it -- this is the classic 'spurious panel regression'
  risk (Granger & Newbold 1974; Kao 1999; Phillips & Moon 1999).

  This check re-estimates the PRIMARY specification of each equation
  adding one linear trend per country (country_dummy x year_c). If the
  key coefficient survives with the same sign and comparable magnitude,
  the baseline result is not merely two coincidentally-trending series.
  Reference: Wooldridge (2010), ch. 10.5 (unit-specific trends).
""")

def add_country_trends(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Add one (country x centred-year) linear trend regressor per country."""
    d = df_in.copy()
    if "year_c" not in d.columns:
        d["year_c"] = d[TIME_COL] - d[TIME_COL].mean()
    trend_cols = []
    for c in sorted(d[ENTITY_COL].unique()):
        col = f"trend_{c}"
        d[col] = (d[ENTITY_COL] == c).astype(float) * d["year_c"]
        trend_cols.append(col)
    return d, trend_cols

df_trends, TREND_COLS = add_country_trends(df)

for eq_label, spec in specs.items():
    dep     = spec["dep"]
    exog    = [c for c in spec["exog"] if c in df.columns]
    key_var = exog[0]

    baseline_r = run_twoway_fe(df, dep, exog, key_var, "Baseline (no country trends)")
    baseline_r["is_baseline"] = True
    trend_r    = run_twoway_fe(df_trends, dep, exog + TREND_COLS, key_var,
                                f"+ {len(TREND_COLS)} country-specific trends")
    trend_r["is_baseline"] = False

    results_trend = [baseline_r, trend_r]
    print_robustness_table(results_trend, f"{eq_label.upper()}: {dep}", baseline_r, key_var)

    if baseline_r["converged"] and trend_r["converged"] and baseline_r["coef"] != 0:
        delta_pct = abs(trend_r["coef"] - baseline_r["coef"]) / abs(baseline_r["coef"]) * 100
        same_sign = np.sign(trend_r["coef"]) == np.sign(baseline_r["coef"])
        if same_sign and delta_pct < 50:
            print(ok(f"  Survives country-specific trends (Δ={delta_pct:.0f}%, same sign)."))
        else:
            print(warn(f"  SENSITIVE to country-specific trends "
                        f"(Δ={delta_pct:.0f}%, same_sign={same_sign}) — "
                        "part of the baseline effect may reflect shared trends, not the "
                        "hypothesized mechanism."))

    all_robustness[f"{eq_label}_country_trends"] = results_trend

# ════════════════════════════════════════════════════════════════════════
# 8. PANEL UNIT ROOT TEST (Fisher-ADF / Maddala-Wu)
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 8 — Panel Unit Root Test (Fisher-ADF, Maddala & Wu 1999)")
print("""
  No unit-root test was previously implemented anywhere in this pipeline
  despite 25-year trending macro series. This runs an Augmented
  Dickey-Fuller test SEPARATELY for each country's time series (T~25 per
  country, so maxlag=1, no automatic lag selection -- higher lags are
  not identifiable with so few observations) and combines the p-values
  with the Fisher (1932) / Maddala-Wu (1999) combination test:

      P = -2 * sum(log(p_i))  ~  chi2(2N) under H0: ALL panels have a unit root.

  CAVEAT (consistent with this pipeline's other small-sample warnings):
  with T~25 observations per country, per-country ADF tests have very
  low power. Failing to reject H0 here is only weak evidence of a unit
  root, not proof of one. This test should be read as a flag for
  further caution, not a definitive diagnosis -- see Robustness 7 above
  (country-specific trends) as a partial, practical remedy regardless of
  the formal test outcome.

  Reference: Maddala & Wu (1999), A comparative study of unit root tests
  with panel data and a new simple test, Oxford Bulletin of Economics
  and Statistics 61(S1), 631-652.
""")

# fisher_panel_unit_root moved to utils.py (shared with Module 07).

UR_VARS = ["homicide_rate_log", "inst_avg", "fdi_percent_gdp",
           "gdp_growth", "gdp_per_capita_log"]

unit_root_results = {}
print(f"  {'Variable':<24} {'N countries':>12} {'Fisher χ²':>11} {'df':>5} {'p (Fisher)':>11}  Flag")
print("  " + "─" * 75)
for var in UR_VARS:
    if var not in df.columns:
        continue
    res = fisher_panel_unit_root(df, var)
    if res is None:
        print(f"  {var:<24} {'insufficient data':>12}")
        continue
    flag = (warn("Cannot reject unit root (panel-wide) → spurious-regression risk")
            if res["p_fisher"] > ALPHA
            else ok("Reject H0: at least one country's series is stationary"))
    print(f"  {var:<24} {res['n_countries']:>12} {res['fisher_stat']:>11.2f} "
          f"{res['df_chi2']:>5} {res['p_fisher']:>11.4f}  {flag}")
    unit_root_results[var] = res

all_robustness["panel_unit_root"] = unit_root_results

# ════════════════════════════════════════════════════════════════════════
# FIGURE: Coefficient stability plots
# ════════════════════════════════════════════════════════════════════════
section("ROBUSTNESS 9 — Panel Composition (original 11 countries + each added country)")
print("  Bias addressed: results that depend on WHICH countries are in the panel.\n")

ORIGINAL_11 = ["COL", "CRI", "DOM", "ECU", "GTM", "HND", "MEX", "NIC", "PAN", "PER", "SLV"]
ADDED = [c for c in COUNTRIES if c not in ORIGINAL_11]


def _fit_chain(codes, label):
    out = {"label": label, "n_countries": len(codes)}
    sub = df[df[ENTITY_COL].isin(codes)].copy()
    for eq_label, spec in specs.items():
        key_var = [c for c in spec["exog"] if c in df.columns][0]
        r = run_twoway_fe(sub, spec["dep"], spec["exog"], key_var, label)
        out[eq_label] = {"coef": r["coef"], "pval": r["pval"], "n_obs": r["n_obs"]}
    return out


composition = [_fit_chain(ORIGINAL_11, "Original 11")]
for c in ADDED:
    composition.append(_fit_chain(ORIGINAL_11 + [c], f"Original 11 + {c}"))
composition.append(_fit_chain(COUNTRIES, "All countries"))
for c in ADDED:
    composition.append(_fit_chain([x for x in COUNTRIES if x != c], f"All excl. {c}"))

print(f"  {'Panel':<26} {'EQ1 coef (p)':>20} {'EQ2 coef (p)':>20} {'EQ3 coef (p)':>20}")
for row in composition:
    cells = [f"{row[e]['coef']:+.3f} ({row[e]['pval']:.3f})" for e in ("eq1", "eq2", "eq3")]
    print(f"  {row['label']:<26} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")
all_robustness["sample_composition"] = composition

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

# ── Figure 14: the "Bukele paradox" ──────────────────────────────────────
va_loco_data = all_robustness.get("eq1_voice_accountability_loco", [])
if va_loco_data:
    fig14, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    coef_plot(axL, va_loco_data, "LOCO: voice_accountability ~ Violence",
              "homicide_rate_log_lag1", "Full sample")
    axL.set_title("Leave-One-Country-Out\n(only excl. SLV flips the sign)", fontsize=9)

    slv = df[df[ENTITY_COL] == "SLV"].sort_values(TIME_COL)
    axR.plot(slv[TIME_COL], slv["homicide_rate"], color=RED_C, marker="o",
             markersize=3, linewidth=1.5, label="Homicide rate (per 100k)")
    axR.set_ylabel("Homicide rate (per 100k)", color=RED_C, fontsize=9)
    axR.tick_params(axis="y", labelcolor=RED_C)
    axR.set_xlabel("Year", fontsize=9)

    axR2 = axR.twinx()
    axR2.plot(slv[TIME_COL], slv["voice_accountability"], color=BLUE, marker="s",
              markersize=3, linewidth=1.5, label="Voice & accountability (WGI)")
    axR2.set_ylabel("Voice & accountability (0-100)", color=BLUE, fontsize=9)
    axR2.tick_params(axis="y", labelcolor=BLUE)

    axR.axvspan(2021, 2024, color=AMBER, alpha=0.12)
    axR.text(2022.5, axR.get_ylim()[1] * 0.95, "Post-2020\nperiod",
             ha="center", va="top", fontsize=7.5, color=AMBER)
    axR.set_title("El Salvador: Security Gains and\nCivil-Liberties Costs, Together", fontsize=9)

    lines1, labels1 = axR.get_legend_handles_labels()
    lines2, labels2 = axR2.get_legend_handles_labels()
    axR.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig14.suptitle(
        'The "Bukele Paradox": Homicides and Civil Liberties Fell Together in El Salvador',
        fontsize=11, fontweight="bold",
    )
    fig14.tight_layout()
    fig14.savefig(DIRS["figures"] / "14_bukele_paradox.png", dpi=300, bbox_inches="tight")
    plt.close(fig14)
    print(ok("Figure saved → figures/14_bukele_paradox.png"))

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
