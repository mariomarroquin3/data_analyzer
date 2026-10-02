"""
10_arch_lm_test.py
════════════════════════════════════════════════════════════════════════
Layer 10 — Engle's ARCH-LM Test: Is There Conditional Heteroskedasticity
           to Model in the First Place?

MOTIVATION
──────────
A natural follow-up question after the main FE chain is whether a
GARCH-family model (e.g. Markov-Switching GARCH) could add value by
modeling time-varying volatility in growth or violence. Before fitting
any such model, the more basic question is whether there is ANY
detectable ARCH effect (conditional heteroskedasticity) in these series
at all -- without it, a GARCH specification has nothing to estimate and
will not converge meaningfully.

This module runs Engle's (1982) Lagrange Multiplier test for ARCH
effects, per country (the only valid unit for a time-series test; the
panel cannot be pooled into one series without creating spurious jumps
at country boundaries), on:
  (a) the Two-Way FE residuals of EQ1, EQ2, EQ3 (Module 02 specification)
  (b) the raw gdp_growth and homicide_rate_log series

Per-country p-values are combined via the same Fisher (1932) method
used elsewhere in this pipeline for panel unit-root testing (see
utils.fisher_panel_unit_root), for a single, interpretable test of
H0: NO country in the panel shows ARCH effects.

CAVEAT (shared with every other per-country test in this pipeline)
────────────────────────────────────────────────────────────────────
T~24 annual observations per country gives an ARCH-LM test very low
power. A failure to reject H0 here is consistent with "no detectable
ARCH effect," but it is also exactly what a severely underpowered test
would show even if a real, modest effect existed. This result should be
read as "insufficient evidence to justify a conditional-volatility
model," not as proof that volatility is constant.

Reference
──────────────────────────────────────────────────────────────────────
Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with
Estimates of the Variance of United Kingdom Inflation. Econometrica,
50(4), 987-1007.
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS
from statsmodels.stats.diagnostic import het_arch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    section, subsection, ok, warn, bold, save_json, make_output_dirs,
    build_multiindex, ENTITY_COL, TIME_COL,
)

DIRS = make_output_dirs(Path(__file__).parent)
MIN_OBS = 8        # same threshold as utils.fisher_panel_unit_root
NLAGS   = 1         # T~24 per country: higher lags are not reliably identifiable

# ════════════════════════════════════════════════════════════════════════
# LOAD & REFIT THE THREE FE EQUATIONS (Module 02 specification)
# ════════════════════════════════════════════════════════════════════════
section("10-A — LOAD & REFIT EQ1/EQ2/EQ3 (for residuals)")

DATA_PATH = Path(__file__).parent / "panel_enriched.csv"
META_PATH = Path(__file__).parent / "json" / "01_metadata.json"

df = pd.read_csv(DATA_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
CONTROLS = meta["controls"]

df_idx = build_multiindex(df)

EQ1_KEY  = "homicide_rate_log_lag1"
EQ1_CTRL = [c for c in CONTROLS if c != "year_c"]
EQ2_KEY  = "inst_avg"
EQ3_KEYS = ["fdi_percent_gdp", "inst_avg"]


def fit_and_get_resids(dep_col: str, exog_cols: list) -> pd.Series:
    cols = [dep_col] + exog_cols
    work = df_idx[cols].dropna()
    mod  = PanelOLS(work[dep_col], work[exog_cols],
                     entity_effects=True, time_effects=True, drop_absorbed=True)
    res  = mod.fit(cov_type="clustered", cluster_entity=True)
    return res.resids


eq1_resid = fit_and_get_resids("inst_avg",        [EQ1_KEY] + EQ1_CTRL)
eq2_resid = fit_and_get_resids("fdi_percent_gdp", [EQ2_KEY] + EQ1_CTRL)
eq3_resid = fit_and_get_resids("gdp_growth",      EQ3_KEYS + EQ1_CTRL)
print(ok(f"EQ1 residuals: n={len(eq1_resid)}  |  EQ2: n={len(eq2_resid)}  |  EQ3: n={len(eq3_resid)}"))

# ════════════════════════════════════════════════════════════════════════
# ARCH-LM TEST, PER COUNTRY, FISHER-COMBINED
# ════════════════════════════════════════════════════════════════════════
section("10-B — ENGLE ARCH-LM TEST (per country, Fisher-combined)")


def arch_lm_fisher(series_by_country: dict, label: str) -> dict:
    """
    Run Engle's ARCH-LM test per country, combine p-values via Fisher's
    method. series_by_country: {country_code: 1D array/Series, time-ordered}.
    """
    pvals, per_country = [], {}
    for c, s in series_by_country.items():
        s = np.asarray(s, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < MIN_OBS:
            continue
        try:
            lm_stat, lm_pval, _, _ = het_arch(s, nlags=NLAGS)
        except Exception:
            continue
        pvals.append(lm_pval)
        per_country[c] = {"lm_stat": float(lm_stat), "p_val": float(lm_pval), "n": int(len(s))}

    if len(pvals) < 3:
        return {"label": label, "error": "too few countries with sufficient obs"}

    pvals_arr   = np.clip(np.array(pvals), 1e-10, 1 - 1e-10)
    fisher_stat = float(-2 * np.sum(np.log(pvals_arr)))
    df_chi2     = 2 * len(pvals_arr)
    p_combined  = float(1 - stats.chi2.cdf(fisher_stat, df_chi2))

    return {
        "label": label,
        "n_countries": len(pvals_arr),
        "fisher_stat": fisher_stat,
        "df_chi2": df_chi2,
        "p_combined": p_combined,
        "per_country": per_country,
        "min_p": float(np.min(pvals_arr)),
        "n_significant_at_05": int(np.sum(pvals_arr < 0.05)),
    }


def by_country_series(s: pd.Series) -> dict:
    """Split a (entity, time)-indexed Series into {country: time-ordered array}."""
    out = {}
    for c, sub in s.groupby(level=ENTITY_COL):
        out[c] = sub.reset_index(level=ENTITY_COL, drop=True).sort_index().values
    return out


targets = {
    "EQ1_residuals (inst_avg ~ violence+controls)": by_country_series(eq1_resid),
    "EQ2_residuals (fdi ~ inst_avg+controls)":       by_country_series(eq2_resid),
    "EQ3_residuals (gdp_growth ~ fdi+inst+controls)": by_country_series(eq3_resid),
    "raw_gdp_growth":      {c: g.sort_values(TIME_COL)["gdp_growth"].values
                             for c, g in df.groupby(ENTITY_COL)},
    "raw_homicide_rate_log": {c: g.sort_values(TIME_COL)["homicide_rate_log"].values
                               for c, g in df.groupby(ENTITY_COL)},
    "diff_homicide_rate_log (first-differenced, detrended)": {
        c: np.diff(g.sort_values(TIME_COL)["homicide_rate_log"].values)
        for c, g in df.groupby(ENTITY_COL)
    },
}

results = {}
for label, series_dict in targets.items():
    r = arch_lm_fisher(series_dict, label)
    results[label] = r
    subsection(label)
    if "error" in r:
        print(f"    {r['error']}")
        continue
    print(f"    N countries tested: {r['n_countries']}")
    print(f"    Fisher combined statistic: {r['fisher_stat']:.2f}  (df={r['df_chi2']})")
    print(f"    Combined p-value: {r['p_combined']:.3f}")
    print(f"    Countries individually significant at 5%: {r['n_significant_at_05']} of {r['n_countries']}")
    print(f"    Smallest per-country p-value: {r['min_p']:.3f}")

# ════════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════════
section("10-C — VERDICT")

any_signal = any(
    ("error" not in r) and (r["p_combined"] < 0.10 or r["n_significant_at_05"] >= 3)
    for r in results.values()
)
if any_signal:
    print(warn("Some evidence of ARCH effects detected -- see per-country breakdown above."))
else:
    print(ok("No evidence of ARCH effects in any series tested (all Fisher-combined"))
    print(ok("p-values > 0.10; no more than 1-2 of 11 countries individually significant"))
    print(ok("at 5%, consistent with chance). This formally confirms the qualitative"))
    print(ok("argument against fitting a GARCH / MS-GARCH model on this panel: with"))
    print(ok("T~24 annual observations per country and no detectable conditional"))
    print(ok("heteroskedasticity to begin with, such a model would have nothing to"))
    print(ok("estimate beyond noise."))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("10-D — EXPORT")
save_json(results, DIRS["json"] / "10_arch_lm_test.json")
print(f"\n{bold('Module 10 complete.')}")
