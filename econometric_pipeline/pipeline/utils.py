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
