"""
01_data_preparation.py
════════════════════════════════════════════════════════════════════════
Layer 1 — Data Validation & Institution Index Construction

Steps
-----
1.  Load and validate the panel (balance, missingness, types).
2.  Construct the PRIMARY institution index:
      standardised average of rule_of_law, control_corruption,
      political_stability (z-scored before averaging).
3.  Construct the ROBUSTNESS institution index:
      PCA on the same three z-scored variables.
      Validated with KMO and Bartlett's sphericity test.
4.  Centre the time trend to mitigate multicollinearity.
5.  Within/between variance decomposition.
6.  Publication-quality summary statistics and correlation heatmap.
7.  Export enriched dataset and metadata JSON.

Literature
----------
Kaiser (1970) — KMO measure of sampling adequacy.
Bartlett (1954) — test of sphericity.
Wooldridge (2010) — within/between decomposition, ch. 10.
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── project imports ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, build_multiindex, within_variance_share,
    ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
SEED = 42

# ════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "panel_ready_for_modeling.csv"

# Variables that constitute the institution index
INST_VARS = ["rule_of_law", "control_corruption", "political_stability"]

# Controls used downstream (validated here)
CONTROLS = [
    "gdp_per_capita_log",
    "inflation",
    "exports_percent_gdp",
    "population_log",
    "unemployment",
]

# Pre-computed lag columns present in the dataset
LAG_COLS = [
    "homicide_rate_log_lag1",
    "homicide_rate_log_lag2",
    "homicide_rate_log_lag3",
    "homicide_rate_lag1",
    "homicide_rate_lag2",
    "homicide_rate_lag3",
]

# ════════════════════════════════════════════════════════════════════════
# 1.  LOAD & VALIDATE
# ════════════════════════════════════════════════════════════════════════
section("01 — DATA LOAD & VALIDATION")

assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)
df = df.sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)

COUNTRIES = sorted(df[ENTITY_COL].unique().tolist())
YEARS     = sorted(df[TIME_COL].unique().tolist())
N         = len(COUNTRIES)
T_total   = len(YEARS)
N_OBS     = len(df)

print(ok(f"Panel loaded: {N_OBS} observations"))
print(ok(f"Countries (N={N}): {', '.join(COUNTRIES)}"))
print(ok(f"Years: {YEARS[0]}–{YEARS[-1]} ({T_total} distinct years)"))

# ── Balance check ────────────────────────────────────────────────────────
subsection("Balance check")
obs_per_country = df.groupby(ENTITY_COL)[TIME_COL].agg(["count", "min", "max"])
obs_per_country.columns = ["T_obs", "year_min", "year_max"]
is_balanced = obs_per_country["T_obs"].nunique() == 1

print(obs_per_country.to_string())
if is_balanced:
    print(ok("Panel is balanced."))
else:
    print(warn("Panel is UNBALANCED — accounting for this in all estimators."))
    missing = []
    for c in COUNTRIES:
        country_years = set(df[df[ENTITY_COL] == c][TIME_COL].tolist())
        full_years    = set(range(YEARS[0], YEARS[-1] + 1))
        gap = full_years - country_years
        if gap:
            missing.append(f"  {c}: missing years {sorted(gap)}")
    if missing:
        print("  Missing country-year cells:")
        for m in missing: print(m)

# ── Column validation ────────────────────────────────────────────────────
subsection("Column validation")
required = INST_VARS + CONTROLS + LAG_COLS + [
    "homicide_rate", "homicide_rate_log",
    "fdi_percent_gdp", "gdp_growth",
    "tourist_arrivals_log", "trade_percent_gdp",
    "time_trend",
]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    print(err(f"Missing columns: {missing_cols}"))
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")
else:
    print(ok(f"All {len(required)} required columns present."))

# ── Missingness map ──────────────────────────────────────────────────────
subsection("Missingness")
miss = df[required].isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss):
    print(warn("Missing values:"))
    for col, n in miss.items():
        pct = n / N_OBS * 100
        print(f"    {col:<35} {n:>4} ({pct:.1f}%)")
else:
    print(ok("No missing values in required columns."))

# ── Numeric types ────────────────────────────────────────────────────────
for col in INST_VARS + CONTROLS + ["homicide_rate_log", "fdi_percent_gdp", "gdp_growth"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ════════════════════════════════════════════════════════════════════════
# 2.  CENTRED TIME TREND
# ════════════════════════════════════════════════════════════════════════
section("02 — CENTRED TIME TREND")

time_mean      = df[TIME_COL].mean()
df["year_c"]   = df[TIME_COL] - time_mean
print(ok(f"year_c = year − {time_mean:.1f}  "
         f"(range: {df['year_c'].min():.0f} to {df['year_c'].max():.0f})"))
print("  Rationale: centring eliminates the large-value component of year that")
print("  inflates the condition number of the design matrix (collinearity).")

# ════════════════════════════════════════════════════════════════════════
# 3.  INSTITUTION INDEX — PRIMARY (STANDARDISED AVERAGE)
# ════════════════════════════════════════════════════════════════════════
section("03 — INSTITUTION INDEX: PRIMARY (STANDARDISED AVERAGE)")

print("  Reference: Kaufmann et al. (2010) — WGI percentile ranks 0–100.")
print("  All three dimensions are first z-scored (global mean=0, SD=1),")
print("  then averaged. This ensures equal weighting regardless of raw scale.")

# Z-score each variable using global (pooled) mean and SD
scaler  = StandardScaler()
z_block = pd.DataFrame(
    scaler.fit_transform(df[INST_VARS].values),
    columns=[f"{v}_z" for v in INST_VARS],
    index=df.index,
)
df = pd.concat([df, z_block], axis=1)

# Simple average of z-scores
df["inst_avg"] = z_block.mean(axis=1)

print(ok(f"inst_avg: mean={df['inst_avg'].mean():.4f}, "
         f"SD={df['inst_avg'].std():.4f}, "
         f"range=[{df['inst_avg'].min():.3f}, {df['inst_avg'].max():.3f}]"))
print("  Higher inst_avg → better governance (all three WGI dimensions).")

# ── Internal consistency: Cronbach's alpha ───────────────────────────────
z_mat  = z_block.dropna().values
k      = z_mat.shape[1]
item_var = z_mat.var(axis=0, ddof=1).sum()
total_var = z_mat.sum(axis=1).var(ddof=1)
cronbach_alpha = (k / (k - 1)) * (1 - item_var / total_var)
alpha_flag = ok(f"α = {cronbach_alpha:.3f} (good internal consistency)") \
    if cronbach_alpha > 0.70 else warn(f"α = {cronbach_alpha:.3f} (low internal consistency)")
print(f"  Cronbach's α (internal consistency of index): {alpha_flag}")

# ════════════════════════════════════════════════════════════════════════
# 4.  INSTITUTION INDEX — ROBUSTNESS (PCA)
# ════════════════════════════════════════════════════════════════════════
section("04 — INSTITUTION INDEX: ROBUSTNESS (PCA)")

print("  PCA operates on the same three z-scored variables.")
print("  PC1 is retained. Sign is aligned so higher = better governance.")

z_matrix = z_block.dropna().values

# KMO measure of sampling adequacy
# Reference: Kaiser (1970) Psychological Bulletin 74(6), 401–404.
def kmo_measure(X: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.
    KMO > 0.6 is considered adequate for PCA/factor analysis.
    """
    from itertools import combinations
    corr  = np.corrcoef(X, rowvar=False)
    corr_inv = np.linalg.pinv(corr)
    # Partial correlation matrix
    diag  = np.diag(corr_inv)
    partial = np.zeros_like(corr_inv)
    for i, j in combinations(range(X.shape[1]), 2):
        partial[i, j] = partial[j, i] = (
            -corr_inv[i, j] / np.sqrt(diag[i] * diag[j])
        )
    # KMO = sum(r²) / (sum(r²) + sum(partial²)) excluding diagonal
    np.fill_diagonal(corr,    0)
    np.fill_diagonal(partial, 0)
    r2  = (corr    ** 2).sum()
    p2  = (partial ** 2).sum()
    kmo_overall = r2 / (r2 + p2)
    kmo_per_var = np.array([
        (corr[:, i] ** 2).sum() /
        ((corr[:, i] ** 2).sum() + (partial[:, i] ** 2).sum())
        for i in range(X.shape[1])
    ])
    return float(kmo_overall), kmo_per_var

# Re-import Tuple for the function above
from typing import Tuple

kmo_val, kmo_per = kmo_measure(z_matrix)
kmo_flag = ok(f"KMO = {kmo_val:.3f} ≥ 0.6 — adequate for PCA") \
    if kmo_val >= 0.6 else warn(f"KMO = {kmo_val:.3f} < 0.6 — marginal adequacy")
print(f"  {kmo_flag}")
for i, var in enumerate(INST_VARS):
    print(f"    {var:<30} KMO = {kmo_per[i]:.3f}")

# Bartlett's test of sphericity
# H0: correlation matrix is identity (no correlations) → reject = PCA meaningful
# Reference: Bartlett (1954) Annals of Mathematical Statistics 25(3), 604–607.
n_obs_pca = z_matrix.shape[0]
corr_mat  = np.corrcoef(z_matrix, rowvar=False)
chi2_stat = -(n_obs_pca - 1 - (2 * k + 5) / 6) * np.log(np.linalg.det(corr_mat))
chi2_df   = k * (k - 1) / 2
chi2_pval = 1 - stats.chi2.cdf(chi2_stat, chi2_df)
bart_flag = ok(f"χ²({chi2_df:.0f})={chi2_stat:.1f}, p={chi2_pval:.4f} — reject H0, PCA appropriate") \
    if chi2_pval < 0.05 else warn(f"Bartlett test: p={chi2_pval:.4f} — fail to reject sphericity")
print(f"  Bartlett sphericity: {bart_flag}")

# Fit PCA
pca  = PCA(n_components=len(INST_VARS), random_state=SEED)
pca.fit(z_matrix)

# Align PC1: higher score = better governance
# Use rule_of_law loading as reference (should be positive)
pc1_scores = pca.transform(z_matrix)[:, 0]
if pca.components_[0][INST_VARS.index("rule_of_law")] < 0:
    pca.components_[0] *= -1
    pc1_scores *= -1

# Only assign to rows where all inst vars are non-null
valid_mask = df[INST_VARS].notna().all(axis=1)
df.loc[valid_mask, "inst_pca"] = pc1_scores

var_exp = pca.explained_variance_ratio_
print(ok(f"PC1 explains {var_exp[0]:.1%} of variance in the three WGI dimensions."))
print(f"  PC2: {var_exp[1]:.1%}, PC3: {var_exp[2]:.1%}")

print("\n  PCA loadings (PC1):")
for i, var in enumerate(INST_VARS):
    bar = "█" * int(abs(pca.components_[0][i]) * 25)
    print(f"    {var:<30} {pca.components_[0][i]:>7.4f}  {bar}")

# Correlation between primary and robustness indices
corr_indices = df[["inst_avg", "inst_pca"]].corr().iloc[0, 1]
corr_flag = ok(f"Correlation(inst_avg, inst_pca) = {corr_indices:.4f} — high convergence") \
    if abs(corr_indices) > 0.9 else warn(f"Correlation = {corr_indices:.4f} — indices diverge")
print(f"\n  {corr_flag}")

# ════════════════════════════════════════════════════════════════════════
# 5.  WITHIN / BETWEEN VARIANCE DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════
section("05 — WITHIN / BETWEEN VARIANCE DECOMPOSITION")

print("  Reference: Wooldridge (2010), Econometric Analysis of Cross Section")
print("  and Panel Data, 2nd ed., ch. 10.")
print("  'Within' variance is the variation exploited by Fixed Effects.\n")

decomp_vars = [
    "homicide_rate_log", "inst_avg", "inst_pca",
    "fdi_percent_gdp", "gdp_growth", "gdp_per_capita_log",
]

print(f"  {'Variable':<28} {'Total SD':>10} {'Between SD':>11} {'Within SD':>10} {'Within %':>10}")
print("  " + "─" * 75)
decomp_results = {}
for var in decomp_vars:
    if var not in df.columns: continue
    total_sd   = df[var].std()
    between_sd = df.groupby(ENTITY_COL)[var].mean().std()
    within_sd  = (df[var] - df.groupby(ENTITY_COL)[var].transform("mean")).std()
    within_pct = within_sd / total_sd * 100 if total_sd > 0 else np.nan
    decomp_results[var] = {
        "total_sd": total_sd, "between_sd": between_sd,
        "within_sd": within_sd, "within_pct": within_pct,
    }
    flag = ok("") if within_pct > 20 else warn("")
    print(f"  {var:<28} {total_sd:>10.4f} {between_sd:>11.4f} "
          f"{within_sd:>10.4f} {within_pct:>9.1f}%  {flag}")

print("\n  ⚠ Variables with low within % are primarily identified from cross-section.")
print("  Fixed Effects may not efficiently estimate their coefficients.")

# ════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════════
section("06 — SUMMARY STATISTICS")

stat_vars = [
    "homicide_rate", "homicide_rate_log",
    "rule_of_law", "control_corruption", "political_stability",
    "inst_avg", "inst_pca",
    "fdi_percent_gdp", "gdp_growth", "gdp_per_capita_log",
    "inflation", "unemployment", "exports_percent_gdp",
    "population_log", "tourist_arrivals_log",
]
stat_vars = [v for v in stat_vars if v in df.columns]

stats_df = df[stat_vars].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).T
stats_df.index.name = "Variable"
print(stats_df[["count","mean","std","min","10%","50%","90%","max"]].round(3).to_string())

# Save as LaTeX table
latex_tbl = stats_df[["count","mean","std","min","50%","max"]].round(3).to_latex(
    caption="Summary statistics — Central America, Colombia and Dominican Republic panel (2000--2024)",
    label="tab:summary_stats",
    column_format="lrrrrrr",
)
(DIRS["tables"] / "summary_statistics.tex").write_text(latex_tbl)
print(ok("LaTeX summary table saved → tables/summary_statistics.tex"))

# ════════════════════════════════════════════════════════════════════════
# 7.  FIGURES
# ════════════════════════════════════════════════════════════════════════
section("07 — FIGURES")

# ── Figure 1: Country-level time series for key variables ────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Panel Overview — Key Variables by Country (2000–2024)",
    fontsize=12, fontweight="bold"
)

plot_pairs = [
    ("homicide_rate",   "Homicide Rate (per 100,000)",        axes[0, 0]),
    ("inst_avg",        "Institution Index (std. avg.)",       axes[0, 1]),
    ("fdi_percent_gdp", "FDI (% of GDP)",                     axes[1, 0]),
    ("gdp_growth",      "GDP Growth Rate (%)",                 axes[1, 1]),
]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for var, ylabel, ax in plot_pairs:
    if var not in df.columns: continue
    for i, country in enumerate(COUNTRIES):
        sub = df[df[ENTITY_COL] == country].sort_values(TIME_COL)
        ax.plot(sub[TIME_COL], sub[var],
                label=country, color=colors[i % len(colors)],
                marker="o", markersize=2, linewidth=1.2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    if var == "gdp_growth":
        ax.axvline(2009, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="GFC")
        ax.axvline(2020, color="grey", linestyle=":",  linewidth=0.8,
                   alpha=0.6, label="COVID")

axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "01_panel_overview.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved → figures/01_panel_overview.png"))

# ── Figure 2: Institution index — primary vs robustness ──────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
sc = ax2.scatter(
    df["inst_avg"], df["inst_pca"],
    c=[COUNTRIES.index(c) for c in df[ENTITY_COL]],
    cmap="tab10", alpha=0.6, s=20,
)
ax2.set_xlabel("Primary Index (Standardised Average)", fontsize=10)
ax2.set_ylabel("Robustness Index (PCA — PC1)", fontsize=10)
ax2.set_title(
    f"Institution Indices: Primary vs Robustness\n"
    f"Pearson r = {corr_indices:.4f}",
    fontsize=10,
)
cbar = fig2.colorbar(sc, ax=ax2, ticks=range(N))
cbar.ax.set_yticklabels(COUNTRIES, fontsize=7)
fig2.tight_layout()
fig2.savefig(DIRS["figures"] / "02_institution_indices.png",
             dpi=300, bbox_inches="tight")
plt.close(fig2)
print(ok("Figure saved → figures/02_institution_indices.png"))

# ── Figure 3: Correlation heatmap ────────────────────────────────────────
import matplotlib.colors as mcolors
heat_vars = ["homicide_rate_log", "inst_avg", "fdi_percent_gdp",
             "gdp_growth", "gdp_per_capita_log", "inflation", "unemployment"]
heat_vars = [v for v in heat_vars if v in df.columns]
corr_mat  = df[heat_vars].corr()

fig3, ax3 = plt.subplots(figsize=(8, 6))
cmap3 = plt.cm.RdBu_r
im = ax3.imshow(corr_mat.values, cmap=cmap3, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax3, shrink=0.8, label="Pearson r")
ax3.set_xticks(range(len(heat_vars)))
ax3.set_yticks(range(len(heat_vars)))
short = [v.replace("_log","(log)").replace("_percent_gdp","(% GDP)").replace("_"," ")
         for v in heat_vars]
ax3.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax3.set_yticklabels(short, fontsize=8)
for i in range(len(heat_vars)):
    for j in range(len(heat_vars)):
        ax3.text(j, i, f"{corr_mat.values[i,j]:.2f}",
                 ha="center", va="center", fontsize=7,
                 color="white" if abs(corr_mat.values[i,j]) > 0.6 else "black")
ax3.set_title("Pooled Correlation Matrix — Key Variables", fontsize=11)
fig3.tight_layout()
fig3.savefig(DIRS["figures"] / "03_correlation_heatmap.png",
             dpi=300, bbox_inches="tight")
plt.close(fig3)
print(ok("Figure saved → figures/03_correlation_heatmap.png"))

# ════════════════════════════════════════════════════════════════════════
# 8.  SAVE ENRICHED DATASET & METADATA
# ════════════════════════════════════════════════════════════════════════
section("08 — EXPORT")

enriched_path = Path(__file__).parent / "panel_enriched.csv"
df.to_csv(enriched_path, index=False)
print(ok(f"Enriched panel saved → {enriched_path.name}  ({len(df)} rows)"))

meta = {
    "countries":         COUNTRIES,
    "years":             YEARS,
    "N":                 N,
    "T_distinct":        T_total,
    "N_obs":             N_OBS,
    "is_balanced":       bool(is_balanced),
    "time_mean":         float(time_mean),
    "inst_vars":         INST_VARS,
    "controls":          CONTROLS,
    "pca_var_exp_pc1":   float(var_exp[0]),
    "pca_loadings":      dict(zip(INST_VARS, pca.components_[0].tolist())),
    "cronbach_alpha":    float(cronbach_alpha),
    "kmo":               float(kmo_val),
    "corr_indices":      float(corr_indices),
    "within_pct":        {k: float(v["within_pct"]) for k, v in decomp_results.items()},
}
save_json(meta, DIRS["json"] / "01_metadata.json")

print(f"\n{bold('Module 01 complete.')} Enriched dataset and metadata exported.")
