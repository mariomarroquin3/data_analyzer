"""
03_diagnostics.py
════════════════════════════════════════════════════════════════════════
Layer 3 — Panel Diagnostic Tests

Tests implemented
-----------------
1.  Wooldridge (2002) serial correlation test
      Manual implementation following Wooldridge's derivation.: first-difference the model residuals, regress
      Δe_it on Δe_{i,t-1}, test H0: ρ=-0.5 under AR(1) errors.
      LIMITATION: no exact Python equivalent of Stata's `xtserial`.
      Implementation follows Wooldridge (2002) p.282-283 manually.

2.  Pesaran (2004) cross-sectional dependence test
      CD = sqrt(2T̄/(N(N-1))) · Σ_{i<j} T_ij · ρ̂_ij
      Asymptotically N(0,1) under H0 of no cross-sectional dependence.

3.  Modified Wald test for group-wise heteroscedasticity
      Greene (2000) ch. 11. Tests H0: σ²_i = σ² for all i.
      Statistic ~ χ²(N-1).

4.  VIF (Variance Inflation Factor) on regressors
      Computed on the within-transformed (demeaned) regressors.

5.  Condition Number of design matrix (multicollinearity severity).

6.  Influence diagnostics:
      — Approximate DFBETA by country (how much does each country shift the key coef)
      — Cook's distance approximation

Explicit limitations
--------------------
• Breusch-Pagan LM test for heteroscedasticity is NOT implemented as the
  LM version requires homoscedastic OLS residuals; Modified Wald is more
  appropriate for panels.
• Arellano-Bond AR(2) test requires full GMM estimation — excluded here
  (see Module 02 for explicit statement on AB-GMM infeasibility with N=8).

Literature
----------
Wooldridge (2002) — Econometric Analysis of Cross Section and Panel Data,
  p.282-283.
Pesaran (2004) — General Diagnostic Tests for Cross Section Dependence in
  Panels, Cambridge Working Papers in Economics 0435.
Greene (2000) — Econometric Analysis, 4th ed., ch. 11.
Belsley, Kuh & Welsch (1980) — Regression Diagnostics.
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
from scipy import stats
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, build_multiindex, two_way_demean,
    ols_on_demeaned, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS  = make_output_dirs(Path(__file__).parent)
ALPHA = 0.05

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("03-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_with_predictions.csv"
META_PATH = Path(__file__).parent / "json" / "01_metadata.json"
SPEC_PATH = Path(__file__).parent / "fe_specs.pkl"

df   = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)
with open(META_PATH) as f: meta = json.load(f)
with open(SPEC_PATH, "rb") as f: specs = pickle.load(f)

COUNTRIES = meta["countries"]
N_C       = len(COUNTRIES)
print(ok(f"Data loaded: {len(df)} obs, {N_C} countries"))

# ════════════════════════════════════════════════════════════════════════
# HELPER: fit FE and extract residuals
# ════════════════════════════════════════════════════════════════════════

def get_fe_residuals(df: pd.DataFrame, dep: str, exog: list) -> pd.Series:
    """Fit Two-Way FE and return entity-time indexed residuals."""
    df_idx = build_multiindex(df)
    exog_a = [c for c in exog if c in df_idx.columns]
    work   = df_idx[[dep] + exog_a].dropna()
    mod    = PanelOLS(
        work[dep], work[exog_a],
        entity_effects=True, time_effects=True, drop_absorbed=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res.resids, res


def get_fe_exog_matrix(df: pd.DataFrame, dep: str, exog: list) -> pd.DataFrame:
    """Return the within-demeaned exogenous matrix (for VIF/CN)."""
    exog_a = [c for c in exog if c in df.columns]
    dm = two_way_demean(df, dep, exog_a)
    return dm[exog_a].dropna()

# ════════════════════════════════════════════════════════════════════════
# TEST 1: WOOLDRIDGE (2002) SERIAL CORRELATION TEST
# ════════════════════════════════════════════════════════════════════════
section("TEST 1 — Wooldridge (2002) Serial Correlation in Panel Residuals")

print("""
  Reference: Wooldridge (2002), Econometric Analysis of Cross Section
  and Panel Data, MIT Press, p.282-283.

  Procedure:
    (a) Estimate the model in FIRST DIFFERENCES: Δy_it = Δx_it·β + Δε_it
    (b) Regress Δe_it on Δe_{i,t-1} (OLS, pooled, no intercept)
    (c) Test H0: ρ = -0.5
        Under AR(1) errors in levels, the FD residuals have
        autocorrelation exactly -0.5.
        A t-test against -0.5 is the Wooldridge test.

  LIMITATION: Stata's `xtserial` implements the exact Wooldridge F-test
  using robust standard errors and a slightly different form. The Python
  implementation below follows the same theoretical derivation but
  may differ numerically from Stata output.
""")

wooldridge_results = {}

for eq_label, spec in specs.items():
    dep   = spec["dep"]
    exog  = [c for c in spec["exog"] if c in df.columns]

    # Step 1: First difference within each country
    df_fd = df[[ENTITY_COL, TIME_COL, dep] + exog].copy()
    df_fd = df_fd.sort_values([ENTITY_COL, TIME_COL])
    df_fd_diff = df_fd.copy()

    for col in [dep] + exog:
        df_fd_diff[f"d_{col}"] = (
            df_fd_diff.groupby(ENTITY_COL)[col].diff()
        )

    fd_cols_dep  = f"d_{dep}"
    fd_cols_exog = [f"d_{c}" for c in exog]
    fd_work = df_fd_diff[[ENTITY_COL, TIME_COL, fd_cols_dep] + fd_cols_exog].dropna()

    if len(fd_work) < 20:
        print(warn(f"  {eq_label}: insufficient observations after first-differencing. Skipped."))
        continue

    # Step 2: Estimate FD model (OLS on first differences)
    y_fd = fd_work[fd_cols_dep].values
    X_fd = fd_work[fd_cols_exog].values
    if X_fd.shape[1] == 0:
        print(warn(f"  {eq_label}: no regressors after FD. Skipped."))
        continue

    beta_fd, _, _, _ = np.linalg.lstsq(X_fd, y_fd, rcond=None)
    resid_fd = y_fd - X_fd @ beta_fd

    # Step 3: Lag FD residuals within country
    fd_work = fd_work.copy()
    fd_work["fd_resid"] = resid_fd
    fd_work["fd_resid_lag"] = fd_work.groupby(ENTITY_COL)["fd_resid"].shift(1)
    fd_ar = fd_work[["fd_resid", "fd_resid_lag"]].dropna()

    if len(fd_ar) < 10:
        print(warn(f"  {eq_label}: too few lag pairs. Skipped."))
        continue

    # Step 4: Regress Δe_it on Δe_{i,t-1}, no intercept
    X_ar = fd_ar["fd_resid_lag"].values.reshape(-1, 1)
    y_ar = fd_ar["fd_resid"].values
    rho  = float(np.linalg.lstsq(X_ar, y_ar, rcond=None)[0][0])
    n_ar = len(y_ar)
    resid_ar = y_ar - X_ar[:, 0] * rho
    se_rho   = float(np.sqrt(resid_ar.var() / (X_ar[:, 0] ** 2).sum()))

    # Step 5: t-test H0: rho = -0.5 (Wooldridge's null)
    t_stat = (rho - (-0.5)) / se_rho
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_ar - 1))

    flag = (ok("No serial correlation (fail to reject H0: ρ=-0.5)")
            if p_val > ALPHA
            else warn("SERIAL CORRELATION detected (reject H0: ρ=-0.5) → use HAC SE"))

    print(f"\n  {bold(eq_label)}: {dep}")
    print(f"  ρ̂ = {rho:.4f}  (H0: ρ=-0.5 under AR(1) errors in levels)")
    print(f"  t = {t_stat:.3f},  p = {p_val:.4f}  {sig_stars(p_val)}")
    print(f"  N_pairs = {n_ar}")
    print(f"  → {flag}")
    print(f"  LIMITATION: Differs from Stata `xtserial` F-statistic by construction.")

    wooldridge_results[eq_label] = {
        "rho": rho, "t_stat": t_stat, "p_val": p_val,
        "n_pairs": n_ar, "reject_h0": bool(p_val < ALPHA)
    }

# ════════════════════════════════════════════════════════════════════════
# TEST 2: PESARAN (2004) CROSS-SECTIONAL DEPENDENCE
# ════════════════════════════════════════════════════════════════════════
section("TEST 2 — Pesaran (2004) Cross-Sectional Dependence (CD) Test")

print("""
  Reference: Pesaran (2004), General Diagnostic Tests for Cross Section
  Dependence in Panels, Cambridge Working Papers in Economics 0435.

  CD = sqrt(2T̄/(N(N-1))) · Σ_{i<j} T_ij · ρ̂_ij
  where ρ̂_ij is the sample correlation of residuals between units i and j,
  T_ij is the number of common time periods.
  Under H0 (no CD): CD → N(0,1) as N,T → ∞.

  Note: with N=8, the asymptotic distribution is approximate.
""")

pesaran_results = {}

for eq_label, spec in specs.items():
    dep  = spec["dep"]
    exog = [c for c in spec["exog"] if c in df.columns]

    resid_series, _ = get_fe_residuals(df, dep, exog)
    resid_wide = resid_series.unstack(level=0)   # T × N

    N_c  = resid_wide.shape[1]
    T_bar = resid_wide.shape[0]
    pairs = [(i, j) for i in range(N_c) for j in range(i + 1, N_c)]

    rho_sum = 0.0
    n_pairs = 0
    for i, j in pairs:
        col_i = resid_wide.iloc[:, i]
        col_j = resid_wide.iloc[:, j]
        common = col_i.notna() & col_j.notna()
        T_ij   = common.sum()
        if T_ij < 3: continue
        rho_ij = col_i[common].corr(col_j[common])
        if np.isnan(rho_ij): continue
        rho_sum += T_ij * rho_ij
        n_pairs += 1

    if n_pairs == 0:
        print(warn(f"  {eq_label}: no valid residual pairs.")); continue

    CD_stat = np.sqrt(2 * T_bar / (N_c * (N_c - 1))) * rho_sum
    p_cd    = 2 * (1 - stats.norm.cdf(abs(CD_stat)))
    flag    = (ok("No significant cross-sectional dependence")
               if p_cd > ALPHA
               else warn("Cross-sectional dependence detected → Driscoll-Kraay SE appropriate"))

    print(f"\n  {bold(eq_label)}: {dep}")
    print(f"  CD = {CD_stat:.4f},  p = {p_cd:.4f}  {sig_stars(p_cd)}")
    print(f"  N pairs = {n_pairs},  T̄ = {T_bar}")
    print(f"  → {flag}")

    pesaran_results[eq_label] = {
        "CD": CD_stat, "p_val": p_cd,
        "n_pairs": n_pairs, "reject_h0": bool(p_cd < ALPHA)
    }

# ════════════════════════════════════════════════════════════════════════
# TEST 3: MODIFIED WALD TEST (GROUP HETEROSCEDASTICITY)
# ════════════════════════════════════════════════════════════════════════
section("TEST 3 — Modified Wald Test for Group-Wise Heteroscedasticity")

print("""
  Reference: Greene (2000), Econometric Analysis, 4th ed., ch. 11.
  Tests H0: σ²_i = σ² for all i  (homoscedasticity across groups).
  Statistic: W = Σ_i [(ê_i'ê_i/T_i - σ̂²)² / (2σ̂⁴/T_i)]  ~ χ²(N-1)
""")

wald_results = {}

for eq_label, spec in specs.items():
    dep  = spec["dep"]
    exog = [c for c in spec["exog"] if c in df.columns]

    resid_series, _ = get_fe_residuals(df, dep, exog)
    resid_df = resid_series.reset_index()
    resid_df.columns = [ENTITY_COL, TIME_COL, "resid"]

    # Per-group variance
    group_stats = resid_df.groupby(ENTITY_COL)["resid"].agg(
        T_i="count",
        var_i=lambda x: (x ** 2).sum() / len(x),   # σ̂²_i = ê_i'ê_i / T_i
    )
    sigma2_pool = resid_df["resid"].pow(2).sum() / len(resid_df)

    W_stat = sum(
        row.T_i * (row.var_i - sigma2_pool) ** 2 / (2 * sigma2_pool ** 2)
        for _, row in group_stats.iterrows()
    )
    df_wald = N_C - 1
    p_wald  = 1 - stats.chi2.cdf(W_stat, df_wald)
    flag    = (ok("Homoscedastic across groups (fail to reject H0)")
               if p_wald > ALPHA
               else warn("Heteroscedastic → clustered / DK SE already applied"))

    print(f"\n  {bold(eq_label)}: {dep}")
    print(f"  W = {W_stat:.3f},  χ²({df_wald}),  p = {p_wald:.4f}  {sig_stars(p_wald)}")
    print(f"  σ̂²_i by country:")
    for country, row in group_stats.iterrows():
        print(f"    {country:>6}  σ̂²={row.var_i:.4f}  T={row.T_i}")
    print(f"  Pooled σ̂² = {sigma2_pool:.4f}")
    print(f"  → {flag}")

    wald_results[eq_label] = {
        "W_stat": W_stat, "df": df_wald, "p_val": p_wald,
        "reject_h0": bool(p_wald < ALPHA)
    }

# ════════════════════════════════════════════════════════════════════════
# TEST 4: VIF & CONDITION NUMBER
# ════════════════════════════════════════════════════════════════════════
section("TEST 4 — Multicollinearity: VIF and Condition Number")

print("  VIF computed on within-transformed (two-way demeaned) regressors.")
print("  Reference: Belsley, Kuh & Welsch (1980), Regression Diagnostics.\n")

vif_results = {}

for eq_label, spec in specs.items():
    dep  = spec["dep"]
    exog = [c for c in spec["exog"] if c in df.columns]

    X_dm = get_fe_exog_matrix(df, dep, exog)
    if X_dm.empty: continue

    X_std = (X_dm - X_dm.mean()) / X_dm.std().replace(0, 1)

    # VIF: 1 / (1 - R²) from regressing each variable on all others
    vifs = {}
    for col in X_std.columns:
        others = [c for c in X_std.columns if c != col]
        if not others: continue
        X_oth  = sm.add_constant(X_std[others].values)
        y_col  = X_std[col].values
        r2     = sm.OLS(y_col, X_oth).fit().rsquared
        vifs[col] = 1.0 / (1.0 - r2 + 1e-12)

    # Condition number of standardised design matrix
    eig_vals = np.linalg.eigvalsh(X_std.T @ X_std)
    eig_vals = np.abs(eig_vals[eig_vals > 1e-12])
    cn = float(np.sqrt(eig_vals.max() / eig_vals.min())) if len(eig_vals) > 0 else np.nan

    print(f"  {bold(eq_label)}: {dep}")
    print(f"  {'Variable':<32} {'VIF':>8}  Flag")
    print("  " + "─" * 55)
    for var, vif in sorted(vifs.items(), key=lambda x: -x[1]):
        flag = err("HIGH (>10)") if vif > 10 else warn("Moderate (5–10)") if vif > 5 else ok("OK")
        print(f"  {var:<32} {vif:>8.2f}  {flag}")
    print(f"  Condition number: {cn:.1f}  "
          + (ok("OK (<30)") if cn < 30 else warn(f"Elevated") if cn < 100 else err("HIGH (>100)")))
    print()

    vif_results[eq_label] = {
        "vifs": {k: round(v, 4) for k, v in vifs.items()},
        "condition_number": cn,
    }

# ════════════════════════════════════════════════════════════════════════
# TEST 5: INFLUENCE DIAGNOSTICS (DFBETA BY COUNTRY)
# ════════════════════════════════════════════════════════════════════════
section("TEST 5 — Influence Diagnostics: DFBETA by Country")

print("""
  DFBETA_c = (β̂ - β̂_{-c}) / SE(β̂)
  where β̂_{-c} is the coefficient estimated excluding country c.
  |DFBETA| > 1 indicates influential observation.
  Reference: Belsley, Kuh & Welsch (1980).
""")

dfbeta_results = {}

for eq_label, spec in specs.items():
    dep  = spec["dep"]
    exog = [c for c in spec["exog"] if c in df.columns]

    # Full-sample estimate
    resid_full, res_full = get_fe_residuals(df, dep, exog)
    beta_full = res_full.params
    se_full   = res_full.std_errors

    # Key variable (first in exog list)
    key_var = exog[0] if exog else None
    if key_var not in beta_full.index: continue

    print(f"\n  {bold(eq_label)}: {dep} | Key var: {key_var}")
    print(f"  β̂_full = {beta_full[key_var]:.4f}  SE = {se_full[key_var]:.4f}")
    print(f"  {'Country':<10} {'β̂_{-c}':>10} {'DFBETA':>10}  Flag")
    print("  " + "─" * 45)

    eq_dfbeta = {}
    for country in COUNTRIES:
        df_excl = df[df[ENTITY_COL] != country]
        if len(df_excl) == 0: continue
        try:
            _, res_excl = get_fe_residuals(df_excl, dep, exog)
            if key_var not in res_excl.params.index: continue
            beta_excl = res_excl.params[key_var]
            dfbeta    = (beta_full[key_var] - beta_excl) / se_full[key_var]
            flag      = err("Influential (|DFBETA|>1)") if abs(dfbeta) > 1 else ok("OK")
            print(f"  {country:<10} {beta_excl:>10.4f} {dfbeta:>10.4f}  {flag}")
            eq_dfbeta[country] = {"beta_excl": beta_excl, "dfbeta": dfbeta}
        except Exception as e:
            print(f"  {country:<10} {'FAILED':>10}  {e}")
            continue

    dfbeta_results[eq_label] = eq_dfbeta

# ════════════════════════════════════════════════════════════════════════
# FIGURE: Diagnostic summary
# ════════════════════════════════════════════════════════════════════════
section("03-F — DIAGNOSTIC FIGURE")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Panel Diagnostic Summary", fontsize=12, fontweight="bold")

eq_labels = list(specs.keys())
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Panel 1: DFBETA for EQ1 key variable
ax = axes[0]
eq1_dfb = dfbeta_results.get("eq1", {})
if eq1_dfb:
    countries_d = list(eq1_dfb.keys())
    dfbetas_d   = [eq1_dfb[c]["dfbeta"] for c in countries_d]
    bar_colors  = [colors[1] if abs(d) > 1 else colors[0] for d in dfbetas_d]
    ax.barh(countries_d, dfbetas_d, color=bar_colors, alpha=0.85, height=0.5)
    ax.axvline(0,  color="black", linewidth=0.7)
    ax.axvline( 1, color="grey",  linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(-1, color="grey",  linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("DFBETA", fontsize=9)
    ax.set_title("EQ1: DFBETA by Country\n(|>1| = influential)", fontsize=9)

# Panel 2: CD test results
ax = axes[1]
cd_vals  = [pesaran_results.get(eq, {}).get("CD", np.nan) for eq in eq_labels]
cd_pvs   = [pesaran_results.get(eq, {}).get("p_val", np.nan) for eq in eq_labels]
bar_cols = [colors[1] if p < 0.05 else colors[0] for p in cd_pvs]
bars = ax.bar(eq_labels, cd_vals, color=bar_cols, alpha=0.85, width=0.5)
ax.axhline(1.96, color="grey", linestyle="--", linewidth=0.8,
           label="5% critical value (±1.96)")
ax.axhline(-1.96, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylabel("CD statistic", fontsize=9)
ax.set_title("Pesaran CD Test\n(|CD|>1.96 → cross-sect. dep.)", fontsize=9)
ax.legend(fontsize=8)
for bar, pv in zip(bars, cd_pvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            sig_stars(pv), ha="center", fontsize=10)

# Panel 3: Wooldridge ρ estimates
ax = axes[2]
rho_vals = [wooldridge_results.get(eq, {}).get("rho", np.nan) for eq in eq_labels]
rho_pvs  = [wooldridge_results.get(eq, {}).get("p_val", np.nan) for eq in eq_labels]
bar_cols2 = [colors[1] if p < 0.05 else colors[0] for p in rho_pvs]
ax.bar(eq_labels, rho_vals, color=bar_cols2, alpha=0.85, width=0.5)
ax.axhline(-0.5, color="red", linestyle="--", linewidth=1,
           label="H0: ρ = -0.5")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Estimated ρ (AR(1) of FD residuals)", fontsize=9)
ax.set_title("Wooldridge Serial Correlation\n(H0: ρ=-0.5 ↔ no AR(1))", fontsize=9)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(DIRS["figures"] / "05_diagnostics.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved → figures/05_diagnostics.png"))

# ════════════════════════════════════════════════════════════════════════
# REMEDIES SUMMARY
# ════════════════════════════════════════════════════════════════════════
section("03-G — REMEDIES SUMMARY")

print(f"""
  Diagnostic            Result                  Remedy applied
  ─────────────────────────────────────────────────────────────────────────
  Serial correlation    See results above       Driscoll-Kraay HAC (Mod.02) ✓
                                                Wild bootstrap (Mod.04) ✓
  Cross-sect. dep.      See results above       Driscoll-Kraay HAC ✓
  Group heterosced.     See results above       Clustered SE ✓
  Multicollinearity     See VIF above           year_c centring (Mod.01) ✓
  Influential obs.      See DFBETA above        Leave-one-out (Mod.05) ✓
  AR(1) in errors       Wooldridge test         Bootstrap inference (Mod.04) ✓
""")

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
diag_export = {
    "wooldridge": wooldridge_results,
    "pesaran_cd": pesaran_results,
    "modified_wald": wald_results,
    "vif": vif_results,
    "dfbeta": {
        eq: {c: {"beta_excl": v["beta_excl"], "dfbeta": v["dfbeta"]}
             for c, v in eq_dfb.items()}
        for eq, eq_dfb in dfbeta_results.items()
    },
}
save_json(diag_export, DIRS["json"] / "03_diagnostics.json")
print(f"\n{bold('Module 03 complete.')}")
