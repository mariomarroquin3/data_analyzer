"""
utils.py
════════════════════════════════════════════════════════════════════════
Shared utilities for the Violence → Institutions → FDI → Growth pipeline.

All modules import from here. Functions are documented with references
to the methodological literature.
════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# ── Publication-quality plot style ──────────────────────────────────────
def set_plot_style() -> None:
    """
    Configure matplotlib for publication-quality output.
    Follows conventions typical of journals such as
    Journal of Development Economics and World Development.
    """
    mpl.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "font.family":         "serif",
        "font.serif":          ["Times New Roman", "DejaVu Serif"],
        "font.size":           10,
        "axes.titlesize":      11,
        "axes.labelsize":      10,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "figure.titlesize":    12,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.5,
        "lines.linewidth":     1.5,
        "patch.edgecolor":     "white",
        "axes.prop_cycle":     mpl.cycler(color=[
            "#2563EB", "#DC2626", "#16A34A", "#D97706",
            "#7C3AED", "#0891B2", "#BE185D", "#065F46"
        ]),
    })


# ── Console formatting ───────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> str: return f"{GREEN}✓{RESET} {msg}"
def warn(msg: str) -> str: return f"{YELLOW}⚠{RESET} {msg}"
def err(msg: str)  -> str: return f"{RED}✗{RESET} {msg}"
def bold(msg: str) -> str: return f"{BOLD}{msg}{RESET}"

def section(title: str) -> None:
    bar = "═" * 65
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{bar}{RESET}")

def subsection(title: str) -> None:
    print(f"\n  {BOLD}── {title} ──{RESET}")

def sig_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

# ── Logger ───────────────────────────────────────────────────────────────
def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ── JSON export ──────────────────────────────────────────────────────────
def save_json(obj: dict, path: Union[str, Path]) -> None:
    """Serialise dict to JSON, converting numpy types to Python natives."""
    def _convert(o):
        if isinstance(o, (np.integer,)):            return int(o)
        if isinstance(o, (np.floating,)):           return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):               return o.tolist()
        if isinstance(o, pd.Series):                return o.to_dict()
        if isinstance(o, pd.DataFrame):             return o.to_dict(orient="records")
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_convert)
    print(ok(f"Saved → {path}"))


# ── Panel utilities ──────────────────────────────────────────────────────
ENTITY_COL = "country_code"
TIME_COL   = "year"

def build_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set (country_code, year) MultiIndex required by linearmodels.
    Validates that neither level contains NaN.
    """
    assert ENTITY_COL in df.columns, f"Missing column: {ENTITY_COL}"
    assert TIME_COL   in df.columns, f"Missing column: {TIME_COL}"
    assert df[ENTITY_COL].notna().all(), "NaN in country_code"
    assert df[TIME_COL].notna().all(),   "NaN in year"
    return df.set_index([ENTITY_COL, TIME_COL])


def within_variance_share(df: pd.DataFrame, var: str) -> float:
    """
    Compute the share of total variance that is within-country variation.
    High within-share → Fixed Effects exploits meaningful time variation.
    """
    total_var   = df[var].var()
    within_var  = (
        df[var] - df.groupby(ENTITY_COL)[var].transform("mean")
    ).var()
    return within_var / total_var if total_var > 0 else np.nan


# ── Two-way within transformation (manual, for bootstrap) ────────────────
def two_way_demean(
    df_long: pd.DataFrame,
    dep_col: str,
    exog_cols: List[str],
    entity_col: str = ENTITY_COL,
    time_col:   str = TIME_COL,
    n_iter:     int = 20,
) -> pd.DataFrame:
    """
    Iterative two-way within transformation (entity + time demeaning).

    Convergence guaranteed because within-group and between-group projections
    are complementary — see Mundlak (1978), Gaure (2013).

    Parameters
    ----------
    df_long   : long-format DataFrame with entity and time columns
    dep_col   : dependent variable column name
    exog_cols : list of regressor column names
    n_iter    : number of alternating projections (20 is conservative)

    Returns
    -------
    DataFrame with demeaned columns (same index as input).
    """
    if entity_col not in df_long.columns or time_col not in df_long.columns:
        if isinstance(df_long.index, pd.MultiIndex):
            index_names = list(df_long.index.names)
            if entity_col in index_names and time_col in index_names:
                df_long = df_long.reset_index()
            elif entity_col in index_names:
                df_long = df_long.reset_index(level=entity_col)
            elif time_col in index_names:
                df_long = df_long.reset_index(level=time_col)

    assert entity_col in df_long.columns, f"Missing required entity column: {entity_col}"
    assert time_col   in df_long.columns, f"Missing required time column: {time_col}"

    cols = [dep_col] + exog_cols
    out  = df_long[cols + [entity_col, time_col]].copy()

    for _ in range(n_iter):
        for col in cols:
            out[col] = out[col] - out.groupby(entity_col)[col].transform("mean")
            out[col] = out[col] - out.groupby(time_col  )[col].transform("mean")

    return out


def ols_on_demeaned(
    df_dm: pd.DataFrame,
    dep_col:   str,
    exog_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    OLS on already within-transformed data.

    Returns
    -------
    beta   : coefficient vector (k,)
    resid  : residual vector (n,)
    df_res : residual degrees of freedom
    """
    mask = df_dm[[dep_col] + exog_cols].notna().all(axis=1)
    Y    = df_dm.loc[mask, dep_col].values
    X    = df_dm.loc[mask, exog_cols].values
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid  = Y - X @ beta
    df_res = n - k         # NOTE: does not subtract FE df (conservative)
    return beta, resid, df_res


# ── Print coefficient table ──────────────────────────────────────────────
def print_coef_table(
    params:    pd.Series,
    std_errs:  pd.Series,
    pvalues:   pd.Series,
    conf_int:  pd.DataFrame,
    title:     str = "",
    key_vars:  Optional[List[str]] = None,
) -> None:
    """Pretty-print a coefficient table with stars and CI."""
    if title:
        print(f"\n  {BOLD}{title}{RESET}")
    hdr = f"  {'Variable':<32} {'Coef':>10} {'SE':>9} {'t':>7} {'p':>7} {'[95% CI]':>22}"
    print(hdr)
    print("  " + "─" * 85)
    for var in params.index:
        hi  = BOLD if (key_vars and var in key_vars) else ""
        se  = std_errs[var]
        t   = params[var] / se if se != 0 else np.nan
        ci  = conf_int.loc[var]
        lo  = ci.iloc[0]
        up  = ci.iloc[1]
        s   = sig_stars(pvalues[var])
        print(
            f"  {hi}{var:<32}{RESET} "
            f"{params[var]:>10.4f} {se:>9.4f} {t:>7.2f} "
            f"{pvalues[var]:>7.3f}{s:<3}  "
            f"[{lo:>7.4f}, {up:>7.4f}]"
        )
    print("  Significance: * p<0.10  ** p<0.05  *** p<0.01")


# ── Normalise the two 02_fe_results.json export shapes ───────────────────
def get_fe_key_stats(fe_eq: dict, key_var: str) -> Dict[str, float]:
    """
    json/02_fe_results.json uses TWO different shapes:
      - EQ1/EQ2 (single key regressor): singular keys
            "coef_key_cl", "se_key_cl", "pval_key_cl",
            "coef_key_dk", "se_key_dk", "pval_key_dk"
      - EQ3 (multiple key regressors): dict-valued, keyed by variable name
            "params_cl", "se_cl", "pvals_cl",
            "params_dk", "se_dk", "pvals_dk"

    Looking up "pval_key_cl" (etc.) directly on an EQ3 dict silently
    returns the .get(...) default (NaN) instead of raising -- this
    produced 'nan' in the printed SE-comparison table and always
    resolved to '?' in the FE-vs-ML convergence table for EQ3.

    This helper normalises both shapes so callers never need to branch
    on which equation they are looking at.
    """
    def _get(dict_key: str, singular_key: str) -> float:
        if singular_key in fe_eq:
            return fe_eq[singular_key]
        return fe_eq.get(dict_key, {}).get(key_var, np.nan)

    return {
        "coef":    _get("params_cl", "coef_key_cl"),
        "se_cl":   _get("se_cl",     "se_key_cl"),
        "pval_cl": _get("pvals_cl",  "pval_key_cl"),
        "se_dk":   _get("se_dk",     "se_key_dk"),
        "pval_dk": _get("pvals_dk",  "pval_key_dk"),
    }


# ── Panel unit-root test (Fisher-ADF / Maddala-Wu) ────────────────────────
def fisher_panel_unit_root(
    df_in: pd.DataFrame, var: str, regression: str = "c",
    entity_col: str = ENTITY_COL, time_col: str = TIME_COL,
) -> Optional[dict]:
    """
    Fisher (1932) / Maddala & Wu (1999) combination panel unit-root test.

    Runs an Augmented Dickey-Fuller test separately for each cross-section
    (maxlag=1, no automatic lag selection -- with T~25 per country, higher
    lags are not reliably identifiable), then combines the per-country
    p-values:  P = -2 * sum(log(p_i)) ~ chi2(2N) under
    H0: ALL cross-sections have a unit root.

    CAVEAT: with T~25 observations per country, per-country ADF tests have
    very low power. Failing to reject H0 is only weak evidence of a unit
    root, not proof of one -- shared by every module that calls this.

    Reference: Maddala & Wu (1999), A comparative study of unit root tests
    with panel data and a new simple test, Oxford Bulletin of Economics
    and Statistics 61(S1), 631-652.

    Returns None if fewer than 3 cross-sections have enough observations.
    """
    countries_local = sorted(df_in[entity_col].unique())
    pvals, per_country = [], {}
    for c in countries_local:
        series = (df_in.loc[df_in[entity_col] == c, [time_col, var]]
                  .dropna().sort_values(time_col)[var].values)
        if len(series) < 8:
            continue
        try:
            stat, pval = adfuller(series, maxlag=1, regression=regression, autolag=None)[:2]
        except Exception:
            continue
        pvals.append(pval)
        per_country[c] = {"adf_stat": float(stat), "p_val": float(pval)}

    if len(pvals) < 3:
        return None
    pvals_arr    = np.clip(np.array(pvals), 1e-10, 1 - 1e-10)
    fisher_stat  = float(-2 * np.sum(np.log(pvals_arr)))
    df_chi2      = 2 * len(pvals_arr)
    p_fisher     = float(1 - stats.chi2.cdf(fisher_stat, df_chi2))
    return {
        "per_country": per_country, "n_countries": len(pvals_arr),
        "fisher_stat": fisher_stat, "df_chi2": df_chi2, "p_fisher": p_fisher,
    }


# ── Webb (2023) weights ──────────────────────────────────────────────────
WEBB_WEIGHTS = np.array([
    -np.sqrt(3 / 2), -1.0, -np.sqrt(1 / 2),
     np.sqrt(1 / 2),  1.0,  np.sqrt(3 / 2),
])
"""
Six-point discrete distribution for wild cluster bootstrap.

Reference: Webb (2023), "Reworking wild bootstrap-based inference for
clustered errors", Canadian Journal of Economics 56(3), 839–858.

Properties: E[w]=0, E[w²]=1, E[w³]=0 (symmetric) — improves size control
over Rademacher weights when G (number of clusters) is small (G < 20).
"""


# ── Fast two-way FE regression with wild cluster bootstrap (FWL) ─────────
def fe_wild_test(
    work: pd.DataFrame,
    dep: str,
    key: str,
    exog: List[str],
    B: int = 999,
    seed: int = 42,
    entity_effects: bool = True,
    time_effects: bool = True,
) -> Dict[str, float]:
    """
    Coefficient on `key` in a fixed-effects regression of `dep` on `exog`
    (`key` must be in `exog`), with cluster-robust (by country) and
    restricted wild cluster bootstrap p-values (Webb weights, null imposed;
    Cameron, Gelbach & Miller, 2008; Webb, 2023).

    `work` must have a (country_code, year) MultiIndex and no missing values
    in dep/exog. Frisch-Waugh-Lovell makes each bootstrap draw O(n):
    with Q the residual-maker of {other regressors, entity dummies, time
    dummies} and x~ = Q x_key, the restricted residual e_r = Q y satisfies
    Q y* = w * e_r for y* = fitted_r + w * e_r, so beta* = x~'(w*e_r) / x~'x~.

    The point estimate equals PanelOLS(entity_effects, time_effects); the
    clustered t uses the CR1 factor G/(G-1) * (n-1)/(n-k), where k counts
    the regressors only, so its standard error is a few percent smaller than
    PanelOLS's. p_wild is (#{|t*| >= |t|} + 1) / (B + 1): it can never be
    below 1/(B+1), so a value equal to that floor means "p <= 1/(B+1)".
    Module 04 reports #{|t*| >= |t|} / B with PanelOLS standard errors, so
    the two agree only up to Monte-Carlo error and these conventions
    (tests/test_pipeline_outputs.py checks the gap stays small).
    """
    ent = work.index.get_level_values(0)
    yr = work.index.get_level_values(1)
    n = len(work)
    y = work[dep].to_numpy(float)
    xk = work[key].to_numpy(float)
    others = [c for c in exog if c != key]
    cols = []
    if others:
        cols.append(work[others].to_numpy(float))
    if entity_effects:
        cols.append(pd.get_dummies(ent).to_numpy(float))
    if time_effects:
        d = pd.get_dummies(yr).to_numpy(float)
        cols.append(d[:, 1:] if entity_effects else d)
    if not cols:
        cols.append(np.ones((n, 1)))
    Z = np.hstack(cols)
    Q = np.eye(n) - Z @ np.linalg.pinv(Z)
    xt = Q @ xk
    sxx = float(xt @ xt)
    if sxx < 1e-12:
        return {"coef": np.nan, "se_cl": np.nan, "t": np.nan, "p_cl": np.nan, "p_wild": np.nan, "n_obs": n}
    codes, uniq = pd.factorize(ent)
    G = len(uniq)
    k = len(exog)
    cr1 = G / (G - 1) * (n - 1) / max(n - k, 1)

    def _se(e):
        s = np.bincount(codes, weights=xt * e, minlength=G)
        se = np.sqrt(cr1 * float(s @ s)) / sxx
        return se

    e_u = Q @ y
    beta = float(xt @ e_u) / sxx
    e_full = e_u - beta * xt
    se = _se(e_full)
    t_obs = beta / se
    p_cl = float(2 * stats.t.sf(abs(t_obs), df=G - 1))

    rng = np.random.default_rng(seed)
    W = WEBB_WEIGHTS[rng.integers(0, 6, size=(G, B))][codes]          # n x B
    ystar = W * e_u[:, None]                                           # = Q y*
    beta_b = (xt @ ystar) / sxx
    resid_b = ystar - beta_b[None, :] * xt[:, None]
    s_b = np.zeros((G, B))
    np.add.at(s_b, codes, xt[:, None] * resid_b)
    se_b = np.sqrt(cr1 * (s_b ** 2).sum(axis=0)) / sxx
    t_b = beta_b / se_b
    p_wild = float((np.sum(np.abs(t_b) >= abs(t_obs)) + 1) / (B + 1))
    return {"coef": beta, "se_cl": se, "t": float(t_obs), "p_cl": p_cl, "p_wild": p_wild, "n_obs": n}


# ── Regression design for a lagged link x -> y (Module 13) ────────────────
def link_design(
    x_col: str,
    y_col: str,
    controls: List[str],
    lag_y: bool = True,
    cause_suffix: str = "__lag1",
) -> Tuple[str, str, List[str]]:
    """
    Dependent column, cause column and regressor list for the lagged link x -> y.

    The cause is `<x_col><cause_suffix>` (the lagged series, or its shuffled
    placebo copy). The lagged-outcome control is ALWAYS the real `<y_col>__lag1`,
    so a placebo that only replaces the cause keeps the persistence control intact.
    """
    ctrl = [c for c in controls if c not in (x_col, y_col)]
    key = f"{x_col}{cause_suffix}"
    exog = [key] + ([f"{y_col}__lag1"] if lag_y else []) + ctrl
    return y_col, key, exog


# ── Country-cluster bootstrap (resample whole countries) ──────────────────
def country_cluster_bootstrap(
    flat: pd.DataFrame,
    stat_fn,
    B: int = 499,
    seed: int = 11,
    entity_col: str = ENTITY_COL,
) -> np.ndarray:
    """
    Resample countries with replacement (all of a country's rows together),
    relabel duplicated draws so each copy is its own entity, and evaluate
    `stat_fn(panel)` on each (country, year)-indexed bootstrap panel.

    `flat` is the un-indexed panel (entity and year as columns). Draws for
    which `stat_fn` returns None / a non-finite value are dropped, so the
    returned array can be shorter than B.
    """
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in flat.groupby(entity_col)}
    names = np.array(sorted(groups))
    out = []
    for _ in range(B):
        pick = rng.choice(names, size=len(names), replace=True)
        parts = []
        for j, c in enumerate(pick):
            g = groups[c].copy()
            g[entity_col] = f"{c}_{j}"
            parts.append(g)
        v = stat_fn(build_multiindex(pd.concat(parts)))
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=float)


# ── Output directories ────────────────────────────────────────────────────
def make_output_dirs(base: Union[str, Path] = ".") -> Dict[str, Path]:
    """Create standard output directory tree."""
    base = Path(base)
    dirs = {
        "figures": base / "figures",
        "tables":  base / "tables",
        "json":    base / "json",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
