"""
02_panel_estimation.py
════════════════════════════════════════════════════════════════════════
Layer 2 — Core Panel Estimation

Estimators
----------
A.  Pooled OLS (benchmark only — inconsistent under FE, reported for
    comparison as in Angrist & Pischke 2009, ch. 5).
B.  Two-Way Fixed Effects (entity + time) — PRIMARY ESTIMATOR.
    Covariance: Driscoll-Kraay (kernel) HAC + clustered by entity.
    Note on CR2: linearmodels v7 does not natively support
    Bell-McCaffrey CR2. CR2 is implemented in Module 04 via
    wild cluster bootstrap which provides correct inference for G=8.
C.  Hausman-type comparison: within vs between coefficients to
    support FE choice.

Equation system (sequential)
-----------------------------
  EQ1: inst_avg_it  = α_i + λ_t + β₁·hom_log_lag1_it + β·Controls + ε₁
  EQ2: fdi_it       = α_i + λ_t + γ₁·inst_avg_it     + γ·Controls + ε₂
  EQ3: gdp_growth_it= α_i + λ_t + δ₁·fdi_it          + δ₂·inst_avg_it
                              + δ·Controls + ε₃

Note on generated regressors
-----------------------------
The sequential OLS approach uses fitted values from EQ1 as a regressor
in EQ2, and EQ2 fitted values in EQ3. This is a SEQUENTIAL OLS procedure,
not a fully efficient 3SLS system. Standard errors in downstream equations
do not account for uncertainty in upstream predictions.
The Wild Cluster Bootstrap in Module 04 partially addresses this by
bootstrapping the FULL system jointly.

This limitation is explicitly flagged in console output.

Literature
----------
Driscoll & Kraay (1998) — kernel HAC for panels with cross-sectional dep.
Angrist & Pischke (2009) — Mostly Harmless Econometrics, ch. 5.
Wooldridge (2010) — Econometric Analysis, ch. 10–11.
Pagan (1984) — generated regressors (limitation of sequential OLS).
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from linearmodels.panel import PanelOLS, PooledOLS, BetweenOLS
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, build_multiindex, ENTITY_COL, TIME_COL,
    print_coef_table, BOLD, RESET,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "panel_enriched.csv"
META_PATH = Path(__file__).parent / "json" / "01_metadata.json"

with open(META_PATH) as f:
    meta = json.load(f)

CONTROLS = meta["controls"]         # from module 01
COUNTRIES = meta["countries"]

# ── Regressors for each equation ─────────────────────────────────────────
# Note: we use pre-computed lag columns from the dataset (not .shift())
# because the panel is unbalanced.
EQ1_KEY  = "homicide_rate_log_lag1"
EQ1_CTRL = [c for c in CONTROLS if c != "year_c"]   # time FE absorbs trend

EQ2_KEY  = "inst_avg"          # actual value (not predicted) in primary spec
EQ2_CTRL = EQ1_CTRL

EQ3_KEYS = ["fdi_percent_gdp", "inst_avg"]
EQ3_CTRL = EQ1_CTRL

# ════════════════════════════════════════════════════════════════════════
# LOAD & VALIDATE
# ════════════════════════════════════════════════════════════════════════
section("02-A — LOAD DATA")

df = pd.read_csv(DATA_PATH)
df = df.sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)

# Check all required columns exist
required = (
    [EQ1_KEY, "inst_avg", "fdi_percent_gdp", "gdp_growth"] +
    EQ1_CTRL + ["year_c"]
)
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
print(ok(f"Data loaded: {len(df)} obs, {df[ENTITY_COL].nunique()} countries"))

# ════════════════════════════════════════════════════════════════════════
# HELPER: fit Two-Way FE with multiple SE estimators
# ════════════════════════════════════════════════════════════════════════

def fit_twoway_fe(
    df_idx:     pd.DataFrame,
    dep_col:    str,
    exog_cols:  list,
    label:      str = "",
) -> dict:
    """
    Fit Two-Way Fixed Effects (entity + time) via linearmodels.PanelOLS.

    Returns a dict with:
      'clustered'  — clustered by entity (conventional, undersized for G=8)
      'driscoll'   — Driscoll-Kraay HAC (robust to cross-sectional dep.)
      'params'     — coefficient estimates (same for both)
      'result_cl'  — full linearmodels result object (clustered)
      'result_dk'  — full linearmodels result object (Driscoll-Kraay)
      'n_obs'      — number of observations used
      'rsq_within' — within R²

    Note: CR2 / wild bootstrap SE are produced in Module 04.
    """
    cols = [dep_col] + exog_cols
    work = df_idx[cols].dropna()
    n    = len(work)

    if n < len(exog_cols) + 10:
        raise ValueError(f"Too few observations ({n}) for {label}")

    mod = PanelOLS(
        work[dep_col],
        work[exog_cols],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )

    # (a) Clustered by entity (conventional — reported as secondary)
    res_cl = mod.fit(cov_type="clustered", cluster_entity=True)

    # (b) Driscoll-Kraay HAC (robust to both serial correlation AND
    #     cross-sectional dependence — preferred given Pesaran CD test)
    #     Bandwidth: Newey-West rule = floor(4*(T/100)^(2/9))
    T_bar     = df_idx.reset_index()[TIME_COL].nunique()
    bandwidth = max(1, int(np.floor(4 * (T_bar / 100) ** (2 / 9))))
    res_dk = mod.fit(
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=bandwidth,
    )

    return {
        "result_cl":  res_cl,
        "result_dk":  res_dk,
        "params":     res_cl.params,
        "n_obs":      n,
        "rsq_within": float(res_cl.rsquared_within),
        "bandwidth":  bandwidth,
    }


def print_fe_results(fit_dict: dict, label: str, key_vars: list) -> None:
    """Print side-by-side: Clustered SE vs Driscoll-Kraay SE."""
    res_cl = fit_dict["result_cl"]
    res_dk = fit_dict["result_dk"]

    print(f"\n  {bold(label)}")
    print(f"  N={fit_dict['n_obs']} | Within R²={fit_dict['rsq_within']:.4f} "
          f"| DK bandwidth={fit_dict['bandwidth']}")
    print(f"\n  {'Variable':<32} "
          f"{'Coef':>10} "
          f"{'SE (Cl.)':>10} {'p(Cl.)':>8} "
          f"{'SE (DK)':>10} {'p(DK)':>8} {'Stars':>6}")
    print("  " + "─" * 90)

    for var in res_cl.params.index:
        coef  = res_cl.params[var]
        se_cl = res_cl.std_errors[var]
        p_cl  = res_cl.pvalues[var]
        se_dk = res_dk.std_errors[var]
        p_dk  = res_dk.pvalues[var]
        hi    = BOLD if var in key_vars else ""
        print(
            f"  {hi}{var:<32}{RESET} "
            f"{coef:>10.4f} "
            f"{se_cl:>10.4f} {p_cl:>8.3f} "
            f"{se_dk:>10.4f} {p_dk:>8.3f} "
            f"{sig_stars(p_dk):>6}"
        )
    print("  Note: Wild cluster bootstrap p-values in Module 04 (preferred for G=8).")


# ════════════════════════════════════════════════════════════════════════
# BENCHMARK: POOLED OLS (for reference only)
# ════════════════════════════════════════════════════════════════════════
section("02-B — BENCHMARK: POOLED OLS")
print(warn("Pooled OLS ignores country fixed effects. Reported for comparison only."))
print("  Reference: Angrist & Pischke (2009) — benchmark before FE.\n")

benchmark_results = {}

for eq_label, dep, exog in [
    ("EQ1: Violence→Institutions",    "inst_avg",         [EQ1_KEY] + EQ1_CTRL + ["year_c"]),
    ("EQ2: Institutions→FDI",         "fdi_percent_gdp",  [EQ2_KEY] + EQ2_CTRL + ["year_c"]),
    ("EQ3: FDI+Inst→GDP Growth",      "gdp_growth",       EQ3_KEYS  + EQ3_CTRL + ["year_c"]),
]:
    df_idx = build_multiindex(df)
    exog_a = [c for c in exog if c in df_idx.columns]
    work   = df_idx[[dep] + exog_a].dropna()

    mod_pool = PooledOLS(work[dep], sm.add_constant(work[exog_a]))
    res_pool = mod_pool.fit(cov_type="clustered", cluster_entity=True)

    print(f"  {eq_label}  (N={len(work)}, R²={res_pool.rsquared:.4f})")
    for var in res_pool.params.index:
        if var == "const": continue
        p = res_pool.pvalues[var]
        print(f"    {var:<32} {res_pool.params[var]:>9.4f}  p={p:.3f}{sig_stars(p)}")
    benchmark_results[eq_label] = {
        "r2": float(res_pool.rsquared),
        "coefs": res_pool.params.to_dict(),
    }

# ════════════════════════════════════════════════════════════════════════
# PRIMARY: TWO-WAY FIXED EFFECTS
# ════════════════════════════════════════════════════════════════════════
section("02-C — PRIMARY: TWO-WAY FIXED EFFECTS")

print("  Estimator: PanelOLS(entity_effects=True, time_effects=True)")
print("  SE reported: (1) Clustered by entity [conventional, G=8 caution]")
print("               (2) Driscoll-Kraay HAC [robust to cross-sect. dep.]")
print("  Preferred inference: Wild Cluster Bootstrap — see Module 04.\n")

df_idx = build_multiindex(df)

# ── EQ1: Violence → Institutional Quality ───────────────────────────────
subsection("EQ1: Violence → Institutional Quality")

eq1_exog = [EQ1_KEY] + [c for c in EQ1_CTRL if c in df_idx.columns]
fe1      = fit_twoway_fe(df_idx, "inst_avg", eq1_exog, "EQ1")
print_fe_results(fe1, "EQ1: homicide_log_lag1 → inst_avg", [EQ1_KEY])

# Generate fitted values for EQ2
fitted1_series = fe1["result_cl"].fitted_values
df["inst_avg_hat"] = np.nan
df.loc[
    df.set_index([ENTITY_COL, TIME_COL]).index.isin(fitted1_series.index), "inst_avg_hat"
] = fitted1_series.values

# ── EQ2: Institutional Quality → FDI ────────────────────────────────────
subsection("EQ2: Institutional Quality → FDI")

# Use observed inst_avg (not predicted) — avoids generated regressor problem
# in primary specification. Predicted values used in robustness.
eq2_exog = [EQ2_KEY] + [c for c in EQ2_CTRL if c in df_idx.columns]
fe2      = fit_twoway_fe(df_idx, "fdi_percent_gdp", eq2_exog, "EQ2")
print_fe_results(fe2, "EQ2: inst_avg → fdi_percent_gdp", [EQ2_KEY])

fitted2_series = fe2["result_cl"].fitted_values
df["fdi_hat"] = np.nan
df.loc[
    df.set_index([ENTITY_COL, TIME_COL]).index.isin(fitted2_series.index), "fdi_hat"
] = fitted2_series.values

# ── EQ3: FDI + Institutions → GDP Growth ────────────────────────────────
subsection("EQ3: FDI + Institutions → GDP Growth")

eq3_exog = EQ3_KEYS + [c for c in EQ3_CTRL if c in df_idx.columns]
fe3      = fit_twoway_fe(df_idx, "gdp_growth", eq3_exog, "EQ3")
print_fe_results(fe3, "EQ3: fdi + inst_avg → gdp_growth", EQ3_KEYS)

# ════════════════════════════════════════════════════════════════════════
# GENERATED REGRESSORS WARNING
# ════════════════════════════════════════════════════════════════════════
section("02-D — METHODOLOGICAL NOTE: GENERATED REGRESSORS")

print(warn("Sequential OLS and the generated regressors problem"))
print("""
  The sequential estimation above uses observed values of inst_avg in EQ2
  and observed fdi in EQ3. This is consistent (each equation is estimated
  independently) but does NOT recover the causal chain from a single
  identified system.

  An alternative using fitted values (inst_avg_hat in EQ2, fdi_hat in EQ3)
  is a 'chain of OLS' approach, which suffers from the 'generated regressors'
  problem (Pagan, 1984, Review of Economic Studies 51(2)):
  — Standard errors in downstream equations are understated because they
    ignore sampling uncertainty in the upstream prediction.

  Correct approaches:
    (a) 3SLS: estimate all three equations simultaneously.
    (b) Causal mediation analysis (Imai et al. 2010) for formal decomposition.
    (c) IV/2SLS for each upstream equation with valid external instruments.

  In this pipeline:
    — Module 02 reports observed-value FE (each equation self-contained).
    — Module 04 bootstrap re-estimates the full system jointly, providing
      honest inference that integrates uncertainty across the chain.
    — IV/2SLS is architecturally prepared but requires external instruments
      (cocaine price index, rainfall anomaly) not present in current dataset.
""")

# ════════════════════════════════════════════════════════════════════════
# HAUSMAN-TYPE COMPARISON (FE vs BE)
# ════════════════════════════════════════════════════════════════════════
section("02-E — HAUSMAN-TYPE FE vs BETWEEN COMPARISON")

print("  Motivation: if FE and Between (cross-sectional average) estimates")
print("  diverge, this suggests omitted time-invariant heterogeneity —")
print("  supporting the use of Fixed Effects.")
print("  Reference: Hausman (1978), Econometrica 46(6), 1251–1271.\n")

hausman_note = []
for eq_label, dep, exog_cols in [
    ("EQ1", "inst_avg",        eq1_exog),
    ("EQ2", "fdi_percent_gdp", eq2_exog),
    ("EQ3", "gdp_growth",      eq3_exog),
]:
    exog_a = [c for c in exog_cols if c in df_idx.columns]
    work   = df_idx[[dep] + exog_a].dropna()

    # Between estimator (unit means)
    be_res = BetweenOLS(work[dep], work[exog_a]).fit(
        cov_type="robust"
    )

    # FE estimates (already computed)
    fe_res_lookup = {"EQ1": fe1, "EQ2": fe2, "EQ3": fe3}
    fe_res_obj    = fe_res_lookup[eq_label]["result_cl"]

    common_vars = [v for v in fe_res_obj.params.index if v in be_res.params.index]
    print(f"  {eq_label}: {dep}")
    print(f"  {'Variable':<30} {'FE coef':>10} {'BE coef':>10} {'Δ':>10}")
    print("  " + "─" * 55)
    for v in common_vars:
        delta = fe_res_obj.params[v] - be_res.params[v]
        flag  = warn(" ← divergence") if abs(delta) > 0.5 * abs(be_res.params[v]) else ""
        print(f"  {v:<30} {fe_res_obj.params[v]:>10.4f} "
              f"{be_res.params[v]:>10.4f} {delta:>10.4f}{flag}")
    print()

# ════════════════════════════════════════════════════════════════════════
# FIGURE: Residual plots for three equations
# ════════════════════════════════════════════════════════════════════════
section("02-F — RESIDUAL FIGURES")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Two-Way Fixed Effects — Residual Diagnostics",
    fontsize=12, fontweight="bold"
)

equations = [
    ("EQ1", fe1, "inst_avg",        "Violence → Institutions"),
    ("EQ2", fe2, "fdi_percent_gdp", "Institutions → FDI"),
    ("EQ3", fe3, "gdp_growth",      "FDI + Inst. → GDP Growth"),
]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for col_i, (eq_label, fe_dict, dep_col, title) in enumerate(equations):
    res    = fe_dict["result_cl"]
    resid  = res.resids
    fitted = res.fitted_values
    entity = resid.index.get_level_values(0)

    # Residuals vs fitted
    ax_top = axes[0, col_i]
    for i, country in enumerate(COUNTRIES):
        mask = entity == country
        ax_top.scatter(
            fitted[mask], resid[mask],
            color=colors[i % len(colors)], s=15, alpha=0.7,
            label=country if col_i == 0 else None,
        )
    ax_top.axhline(0, color="black", linewidth=0.8)
    ax_top.set_xlabel("Fitted values", fontsize=9)
    ax_top.set_ylabel("Residuals", fontsize=9)
    ax_top.set_title(f"{eq_label}: {title}\nResiduals vs Fitted", fontsize=9)

    # Histogram of residuals
    ax_bot = axes[1, col_i]
    ax_bot.hist(resid.values, bins=25, edgecolor="white",
                color=colors[col_i], alpha=0.8)
    from scipy import stats as scipy_stats
    ax_bot.set_xlabel("Residuals", fontsize=9)
    ax_bot.set_ylabel("Count", fontsize=9)
    jb, jb_p = scipy_stats.jarque_bera(resid.dropna())[:2]
    ax_bot.set_title(f"Residual distribution\nJarque-Bera p={jb_p:.3f}", fontsize=9)

if col_i == 0:
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")

fig.tight_layout()
fig.savefig(DIRS["figures"] / "04_fe_residuals.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved → figures/04_fe_residuals.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("02-G — EXPORT")

# Save fitted values for downstream modules
df.to_csv(Path(__file__).parent / "panel_with_predictions.csv", index=False)
print(ok("Dataset with predictions saved → panel_with_predictions.csv"))

results_export = {
    "EQ1": {
        "dep": "inst_avg",
        "key_var": EQ1_KEY,
        "n_obs": fe1["n_obs"],
        "rsq_within": fe1["rsq_within"],
        "coef_key_cl": float(fe1["result_cl"].params[EQ1_KEY]),
        "se_key_cl":   float(fe1["result_cl"].std_errors[EQ1_KEY]),
        "pval_key_cl": float(fe1["result_cl"].pvalues[EQ1_KEY]),
        "coef_key_dk": float(fe1["result_dk"].params[EQ1_KEY]),
        "se_key_dk":   float(fe1["result_dk"].std_errors[EQ1_KEY]),
        "pval_key_dk": float(fe1["result_dk"].pvalues[EQ1_KEY]),
    },
    "EQ2": {
        "dep": "fdi_percent_gdp",
        "key_var": EQ2_KEY,
        "n_obs": fe2["n_obs"],
        "rsq_within": fe2["rsq_within"],
        "coef_key_cl": float(fe2["result_cl"].params[EQ2_KEY]),
        "se_key_cl":   float(fe2["result_cl"].std_errors[EQ2_KEY]),
        "pval_key_cl": float(fe2["result_cl"].pvalues[EQ2_KEY]),
        "coef_key_dk": float(fe2["result_dk"].params[EQ2_KEY]),
        "se_key_dk":   float(fe2["result_dk"].std_errors[EQ2_KEY]),
        "pval_key_dk": float(fe2["result_dk"].pvalues[EQ2_KEY]),
    },
    "EQ3": {
        "dep": "gdp_growth",
        "key_vars": EQ3_KEYS,
        "n_obs": fe3["n_obs"],
        "rsq_within": fe3["rsq_within"],
        "params_cl":  fe3["result_cl"].params.to_dict(),
        "pvals_cl":   fe3["result_cl"].pvalues.to_dict(),
        "params_dk":  fe3["result_dk"].params.to_dict(),
        "pvals_dk":   fe3["result_dk"].pvalues.to_dict(),
    },
}
save_json(results_export, DIRS["json"] / "02_fe_results.json")

# Store model objects for Module 04 (bootstrap needs same spec)
import pickle
with open(Path(__file__).parent / "fe_specs.pkl", "wb") as f:
    pickle.dump({
        "eq1": {"dep": "inst_avg",        "exog": eq1_exog},
        "eq2": {"dep": "fdi_percent_gdp", "exog": eq2_exog},
        "eq3": {"dep": "gdp_growth",      "exog": eq3_exog},
    }, f)
print(ok("FE specifications saved → fe_specs.pkl (for bootstrap module)"))
print(f"\n{bold('Module 02 complete.')}")