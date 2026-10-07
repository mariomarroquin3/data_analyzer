"""
07_cointegration.py
════════════════════════════════════════════════════════════════════════
Layer 7 — Panel Cointegration & Error-Correction Model (ECM)

Motivation
----------
Module 05's Fisher-ADF panel unit-root test found that homicide_rate_log,
inst_avg and gdp_per_capita_log all fail to reject a unit root over this
25-year panel, while fdi_percent_gdp and gdp_growth do not (they reject
it — i.e. they behave like I(0), stationary series). Two individually
non-stationary (I(1)) series can still share a genuine long-run
equilibrium relationship — cointegration — in which case a static
Two-Way FE regression in LEVELS is not necessarily spurious after all,
and a richer Error-Correction Model (ECM) can separate the SHORT-RUN
dynamics of a change in x from the SPEED OF ADJUSTMENT of y back toward
that long-run relationship whenever it drifts away.

Scope: only pairs where BOTH series are I(1) are tested here:
  T1: homicide_rate_log <-> inst_avg           (the EQ1 relationship)
  T2: inst_avg          <-> gdp_per_capita_log  (institutions <-> income)
  T3: homicide_rate_log <-> gdp_per_capita_log  (violence <-> income)

EQ2 (institutions -> FDI) and EQ3 (FDI -> growth) are DELIBERATELY NOT
tested here. fdi_percent_gdp and gdp_growth already reject a unit root
(Module 05) — they are I(0) by construction. Cointegration is a concept
for two variables that individually wander (I(1)) but move together;
pairing an I(1) series with an I(0) series is not a coherent
cointegration question — any linear combination of an I(1) and an I(0)
series is itself I(1), so a residual-based test would trivially find "no
cointegration" and add nothing. This scope restriction is disclosed, not
glossed over.

Method (Engle-Granger-style panel cointegration test)
-------------------------------------------------------
A full Pedroni (1999) panel cointegration test computes 7 separate
statistics (4 "panel" + 3 "group-mean") with country-specific long-run
variance corrections — substantial additional machinery beyond this
project's scope and, per Module 05's own caveat, of doubtful extra value
with only G=18 cross-sections. This module instead implements the
simpler, well-established two-step residual-based approach (Engle &
Granger 1987, extended to panels by Kao 1999 and McCoskey & Kao 1998):

  1. Estimate the long-run relationship via Two-Way FE:
         y_it = alpha_i + lambda_t + beta * x_it + e_it
  2. Test the residuals e_it for a unit root using the SAME Fisher-ADF
     panel test built in Module 05 (Maddala & Wu 1999) — now shared via
     utils.fisher_panel_unit_root().
  3. If the residuals REJECT the unit-root null (i.e. behave as
     stationary), x and y share a genuine long-run equilibrium — proceed
     to the ECM. If not, the levels relationship is treated as
     potentially spurious and no ECM is estimated for that pair — this
     negative result is reported, not hidden.

Error-Correction Model (only for cointegrated pairs)
-------------------------------------------------------
  Delta y_it = alpha_i + lambda_t + phi * ehat_{i,t-1} + gamma * Delta x_it + eps_it

where ehat_{i,t-1} is the LAGGED residual from step 1 (last period's
deviation from long-run equilibrium). phi < 0 and significant means y
corrects part of any deviation each period (the "speed of adjustment").
gamma is the SHORT-RUN effect of a change in x on a change in y —
conceptually distinct from beta, the long-run effect estimated in step 1.

Caveats (consistent with every other small-sample warning in this
pipeline)
-------------------------------------------------------
- G=18 countries, T~25 years is small even for the classic single-series
  Engle-Granger test; the panel extension does not fix a fundamentally
  small sample. Treat every result here as exploratory.
- The Fisher-ADF residual test inherits Module 05's low-power caveat.
- This is a simplified two-step estimator, not a fully efficient panel
  cointegration/ECM system: no correction for cross-sectional dependence
  in the long-run regression, no Pedroni/Westerlund test statistics, and
  standard errors on the ECM are conventional-clustered / Driscoll-Kraay
  only (no wild cluster bootstrap here — a natural next extension,
  flagged rather than silently skipped).

Literature
----------
Engle & Granger (1987) — Co-integration and error correction:
  representation, estimation and testing, Econometrica 55(2), 251-276.
Kao (1999) — Spurious regression and residual-based tests for
  cointegration in panel data, Journal of Econometrics 90(1), 1-44.
McCoskey & Kao (1998) — A residual-based test of the null of
  cointegration in panel data, Econometric Reviews 17(1), 57-84.
Pedroni (1999, 2004) — Panel cointegration test statistics and critical
  values (not implemented here — see "Method" above).
Maddala & Wu (1999) — Fisher-type panel unit-root combination test
  (reused from Module 05 via utils.fisher_panel_unit_root).
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, build_multiindex, fisher_panel_unit_root,
    ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS  = make_output_dirs(Path(__file__).parent)
ALPHA = 0.05

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("07-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_with_predictions.csv"
df = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)
COUNTRIES = sorted(df[ENTITY_COL].unique().tolist())
G = len(COUNTRIES)
print(ok(f"Data loaded: {len(df)} obs, {G} countries"))

print("""
  Scope reminder: this module only tests pairs of series that Module 05's
  Fisher-ADF test found to be I(1) (non-stationary): homicide_rate_log,
  inst_avg, gdp_per_capita_log. fdi_percent_gdp and gdp_growth already
  reject a unit root there (they are I(0)) and are intentionally excluded
  from cointegration testing -- see module docstring.
""")

PAIRS = [
    {"name": "T1: Homicide <-> Institutions", "y": "inst_avg",           "x": "homicide_rate_log"},
    {"name": "T2: Institutions <-> Income",    "y": "inst_avg",           "x": "gdp_per_capita_log"},
    {"name": "T3: Homicide <-> Income",        "y": "gdp_per_capita_log", "x": "homicide_rate_log"},
]

# ════════════════════════════════════════════════════════════════════════
# STEP 1: LONG-RUN RELATIONSHIP (Two-Way FE) + RESIDUAL UNIT-ROOT TEST
# ════════════════════════════════════════════════════════════════════════
section("07-B — STEP 1: LONG-RUN RELATIONSHIP & RESIDUAL COINTEGRATION TEST")

def long_run_regression(df_in: pd.DataFrame, y: str, x: str) -> dict:
    """Two-Way FE long-run regression y_it = a_i + l_t + beta*x_it + e_it."""
    work = df_in[[ENTITY_COL, TIME_COL, y, x]].dropna().copy()
    df_idx = build_multiindex(work)
    mod = PanelOLS(df_idx[y], df_idx[[x]], entity_effects=True, time_effects=True,
                    drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    resid = res.resids.copy()
    resid.name = "resid"
    resid_df = resid.reset_index()  # columns: ENTITY_COL, TIME_COL, resid

    return {
        "beta":       float(res.params[x]),
        "se_cl":      float(res.std_errors[x]),
        "pval_cl":    float(res.pvalues[x]),
        "n_obs":      int(res.nobs),
        "rsq_within": float(res.rsquared_within),
        "resid_df":   resid_df,
    }


cointegration_results = {}

for pair in PAIRS:
    name, y, x = pair["name"], pair["y"], pair["x"]
    subsection(name)
    print(f"  Long-run regression: {y}_it = a_i + l_t + beta*{x}_it + e_it")

    lr = long_run_regression(df, y, x)
    print(f"    beta = {lr['beta']:.4f}   SE(cl) = {lr['se_cl']:.4f}   "
          f"p(cl) = {lr['pval_cl']:.4f}  {sig_stars(lr['pval_cl'])}   N = {lr['n_obs']}")

    ur = fisher_panel_unit_root(lr["resid_df"], "resid")
    if ur is None:
        print(warn("  Residual unit-root test failed (insufficient per-country observations)."))
        cointegration_results[name] = {"y": y, "x": x, "long_run": {k: v for k, v in lr.items() if k != "resid_df"},
                                        "residual_unit_root": None, "cointegrated": False}
        continue

    cointegrated = ur["p_fisher"] < ALPHA
    flag = (ok(f"REJECT unit root in residuals (p={ur['p_fisher']:.4f}) -> evidence of cointegration")
            if cointegrated else
            warn(f"Cannot reject unit root in residuals (p={ur['p_fisher']:.4f}) -> "
                 "NO evidence of cointegration; levels relationship may be spurious"))
    print(f"    Fisher-ADF on residuals: chi2({ur['df_chi2']})={ur['fisher_stat']:.2f}, "
          f"p={ur['p_fisher']:.4f}")
    print(f"    {flag}")

    cointegration_results[name] = {
        "y": y, "x": x,
        "long_run": {k: v for k, v in lr.items() if k != "resid_df"},
        "residual_unit_root": {k: v for k, v in ur.items() if k != "per_country"},
        "cointegrated": bool(cointegrated),
    }

# ════════════════════════════════════════════════════════════════════════
# STEP 2: ERROR-CORRECTION MODEL (only for cointegrated pairs)
# ════════════════════════════════════════════════════════════════════════
section("07-C — STEP 2: ERROR-CORRECTION MODEL (cointegrated pairs only)")

def fit_ecm(df_in: pd.DataFrame, y: str, x: str, resid_df: pd.DataFrame) -> dict:
    """
    Delta y_it = a_i + l_t + phi*ehat_{i,t-1} + gamma*Delta x_it + eps_it
    """
    work = df_in[[ENTITY_COL, TIME_COL, y, x]].merge(
        resid_df, on=[ENTITY_COL, TIME_COL], how="left"
    ).sort_values([ENTITY_COL, TIME_COL])

    work["dy"]        = work.groupby(ENTITY_COL)[y].diff()
    work["dx"]        = work.groupby(ENTITY_COL)[x].diff()
    work["resid_lag"] = work.groupby(ENTITY_COL)["resid"].shift(1)

    ecm_data = work[[ENTITY_COL, TIME_COL, "dy", "dx", "resid_lag"]].dropna()
    df_idx = build_multiindex(ecm_data)

    mod = PanelOLS(df_idx["dy"], df_idx[["resid_lag", "dx"]],
                    entity_effects=True, time_effects=True, drop_absorbed=True)
    res_cl = mod.fit(cov_type="clustered", cluster_entity=True)

    T_bar = df_idx.index.get_level_values(1).nunique()
    bandwidth = max(1, int(np.floor(4 * (T_bar / 100) ** (2 / 9))))
    res_dk = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth)

    return {
        "phi_cl":   float(res_cl.params["resid_lag"]),
        "phi_se_cl":float(res_cl.std_errors["resid_lag"]),
        "phi_p_cl": float(res_cl.pvalues["resid_lag"]),
        "phi_se_dk":float(res_dk.std_errors["resid_lag"]),
        "phi_p_dk": float(res_dk.pvalues["resid_lag"]),
        "gamma_cl":   float(res_cl.params["dx"]),
        "gamma_se_cl":float(res_cl.std_errors["dx"]),
        "gamma_p_cl": float(res_cl.pvalues["dx"]),
        "gamma_se_dk":float(res_dk.std_errors["dx"]),
        "gamma_p_dk": float(res_dk.pvalues["dx"]),
        "n_obs":    int(res_cl.nobs),
        "rsq_within": float(res_cl.rsquared_within),
    }


ecm_results = {}
any_cointegrated = False

for pair in PAIRS:
    name, y, x = pair["name"], pair["y"], pair["x"]
    info = cointegration_results[name]
    subsection(name)

    if not info["cointegrated"]:
        print(warn("  Not cointegrated in Step 1 -- ECM skipped (would be a regression of "
                    "one I(1)-differenced series on an unreliable, non-mean-reverting "
                    "'equilibrium error' -- not meaningful)."))
        continue

    any_cointegrated = True
    work_full = df[[ENTITY_COL, TIME_COL, y, x]].dropna()
    lr_full = long_run_regression(work_full, y, x)
    ecm = fit_ecm(df, y, x, lr_full["resid_df"])

    print(f"  Delta {y}_it = a_i + l_t + phi*ehat_(t-1) + gamma*Delta {x}_it + eps_it")
    print(f"    phi   (speed of adjustment) = {ecm['phi_cl']:.4f}  "
          f"SE(cl)={ecm['phi_se_cl']:.4f} p(cl)={ecm['phi_p_cl']:.4f} "
          f"p(dk)={ecm['phi_p_dk']:.4f}  {sig_stars(ecm['phi_p_cl'])}")
    print(f"    gamma (short-run effect)    = {ecm['gamma_cl']:.4f}  "
          f"SE(cl)={ecm['gamma_se_cl']:.4f} p(cl)={ecm['gamma_p_cl']:.4f} "
          f"p(dk)={ecm['gamma_p_dk']:.4f}  {sig_stars(ecm['gamma_p_cl'])}")
    print(f"    N = {ecm['n_obs']}   Within R^2 = {ecm['rsq_within']:.4f}")

    if ecm["phi_cl"] < 0 and ecm["phi_p_cl"] < ALPHA:
        print(ok(f"  phi < 0 and significant: {y} corrects "
                 f"{abs(ecm['phi_cl'])*100:.1f}% of a deviation from long-run "
                 f"equilibrium with {x} each year."))
    else:
        print(warn("  phi is not both negative and significant at 5% -- weak or no "
                    "error-correction behaviour despite residual stationarity in Step 1."))

    ecm_results[name] = ecm
    cointegration_results[name]["ecm"] = ecm

if not any_cointegrated:
    print(warn("\n  No pair showed evidence of cointegration. This means the apparent "
               "levels relationships between these I(1) series (as estimated by static "
               "Two-Way FE elsewhere in this pipeline) cannot be distinguished from "
               "spurious panel regression with the available data. Report this "
               "explicitly rather than the ECM step -- it is itself a finding."))

# ════════════════════════════════════════════════════════════════════════
# FIGURE: residual series + ECM diagnostics
# ════════════════════════════════════════════════════════════════════════
section("07-D — FIGURES")

n_pairs = len(PAIRS)
fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.5))
if n_pairs == 1:
    axes = [axes]
fig.suptitle(
    "Long-Run Regression Residuals by Country\n"
    "(stationary residuals = evidence of cointegration; Fisher-ADF, Module 05/07)",
    fontsize=11, fontweight="bold",
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, pair in zip(axes, PAIRS):
    name, y, x = pair["name"], pair["y"], pair["x"]
    work_full = df[[ENTITY_COL, TIME_COL, y, x]].dropna()
    lr_full = long_run_regression(work_full, y, x)
    resid_df = lr_full["resid_df"]
    for i, country in enumerate(COUNTRIES):
        sub = resid_df[resid_df[ENTITY_COL] == country].sort_values(TIME_COL)
        ax.plot(sub[TIME_COL], sub["resid"], color=colors[i % len(colors)],
                label=country, linewidth=1.1, marker="o", markersize=2)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    coint = cointegration_results[name]["cointegrated"]
    ax.set_title(f"{name}\n{'Cointegrated' if coint else 'NOT cointegrated'} "
                 f"(p={cointegration_results[name]['residual_unit_root']['p_fisher']:.3f})",
                 fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Residual (equilibrium error)", fontsize=9)

axes[0].legend(fontsize=6, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "11_cointegration_residuals.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/11_cointegration_residuals.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("07-E — EXPORT")
save_json(cointegration_results, DIRS["json"] / "07_cointegration.json")
print(f"\n{bold('Module 07 complete.')}")
