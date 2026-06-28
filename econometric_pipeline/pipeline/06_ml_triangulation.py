"""
06_ml_triangulation.py
════════════════════════════════════════════════════════════════════════
Layer 6 — Machine Learning Triangulation (Exploratory / Non-Causal)

FRAMING CONSTRAINT
───────────────────
ML results are NEVER presented as causal evidence.
They serve to triangulate: if the same variables that are statistically
significant in FE-IV also rank highly in ML feature importance, this
convergence strengthens the narrative.
If they diverge, this signals potential confounders or non-linearities
that the linear FE model may miss.

Cross-Validation Strategy
───────────────────────────
With N=8 countries, ordinary K-Fold leaks country information
(training on part of a country, predicting on another period of the
same country). This produces optimistically biased performance estimates.

CORRECT approach: Leave-One-Country-Out (LOCO) cross-validation.
For each fold, train on all countries EXCEPT one, predict on the
held-out country. This measures true out-of-sample (out-of-country)
predictive power.

Important: G=8 LOCO folds means that all CV performance metrics have
very wide uncertainty (SD across folds is as informative as the mean).
Do not over-interpret absolute values of LOCO R².

Methods
───────
1.  Random Forest (sklearn) with LOCO-CV
2.  Gradient Boosting (sklearn) with LOCO-CV
3.  SHAP TreeExplainer — mean absolute SHAP values (full-sample model)
4.  SHAP cluster bootstrap stability (B=200): resample countries with
    replacement (all years of selected country retained) and compute
    SHAP rankings each time. Reports mean rank, rank SD, and
    probability of appearing in the top-3 and top-5.
5.  ALE (Accumulated Local Effects) plots for key variables.
    ALE is preferred over PDP when predictors are correlated, because
    it conditions on observed neighbourhoods rather than marginalising
    over potentially extrapolated predictor distributions.
    Reference: Apley & Zhu (2020), JRSS-B.
6.  Convergence table: FE coefficient / p-value / significance vs
    ML importance rank.
7.  Wilcoxon signed-rank test comparing RF and GB LOCO fold R².
    Note: with G=8 paired observations, this test has very low power
    and results should be treated as purely descriptive.

Targets modelled separately
───────────────────────────
T1: inst_avg      (violence → institutions mechanism)
T2: fdi_percent_gdp (institutions → FDI)
T3: gdp_growth    (FDI + institutions → growth)

Literature
──────────────────────────────────────────────────────────────────────
Apley & Zhu (2020) — Visualizing the effects of predictor variables
    in black box supervised learning models. JRSS-B.
Athey & Imbens (2019) — Machine learning methods for economists.
Breiman (2001) — Random Forests.
Lundberg & Lee (2017) — SHAP (NeurIPS).
Molnar (2022) — Interpretable Machine Learning, ch. 8–9.
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
from sklearn.impute import SimpleImputer
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("06-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_with_predictions.csv"
META_PATH = Path(__file__).parent / "json" / "01_metadata.json"

df   = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)
with open(META_PATH) as f: meta = json.load(f)

COUNTRIES = meta["countries"]
G         = len(COUNTRIES)
print(ok(f"Data: {len(df)} obs | {G} countries"))
print(f"  CV strategy: Leave-One-Country-Out ({G} folds)")
print(warn("  All ML results are EXPLORATORY. They do not imply causation."))
print(warn(f"  G={G} LOCO folds → CV performance estimates carry wide uncertainty."
           "  Interpret means alongside SDs, not in isolation."))

# ════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════

# Human-readable labels
FEATURE_LABELS = {
    "homicide_rate_log":        "Violence (log homicide rate)",
    "homicide_rate_log_lag1":   "Violence lag 1",
    "homicide_rate_log_lag2":   "Violence lag 2",
    "rule_of_law":              "Rule of Law",
    "control_corruption":       "Control of Corruption",
    "political_stability":      "Political Stability",
    "inst_avg":                 "Institution Index (avg)",
    "fdi_percent_gdp":          "FDI (% GDP)",
    "gdp_per_capita_log":       "GDP per capita (log)",
    "exports_percent_gdp":      "Exports (% GDP)",
    "trade_percent_gdp":        "Trade (% GDP)",
    "inflation":                "Inflation",
    "unemployment":             "Unemployment",
    "population_log":           "Population (log)",
    "tourist_arrivals_log":     "Tourist Arrivals (log)",
    "year_c":                   "Time Trend (centred)",
}

# Task definitions: (target, features, task_name)
TASKS = [
    {
        "target":   "inst_avg",
        "features": [
            "homicide_rate_log_lag1", "homicide_rate_log_lag2",
            "gdp_per_capita_log", "inflation", "unemployment",
            "exports_percent_gdp", "population_log", "year_c",
        ],
        "name": "T1: Violence → Institutions",
        "key_feature": "homicide_rate_log_lag1",
    },
    {
        "target":   "fdi_percent_gdp",
        "features": [
            "inst_avg", "homicide_rate_log_lag1",
            "gdp_per_capita_log", "inflation", "unemployment",
            "exports_percent_gdp", "trade_percent_gdp",
            "tourist_arrivals_log", "year_c",
        ],
        "name": "T2: Institutions → FDI",
        "key_feature": "inst_avg",
    },
    {
        "target":   "gdp_growth",
        "features": [
            "fdi_percent_gdp", "inst_avg", "homicide_rate_log_lag1",
            "gdp_per_capita_log", "inflation", "unemployment",
            "exports_percent_gdp", "year_c",
        ],
        "name": "T3: FDI + Institutions → Growth",
        "key_feature": "fdi_percent_gdp",
    },
]

# ════════════════════════════════════════════════════════════════════════
# LOCO CROSS-VALIDATION FUNCTION
# ════════════════════════════════════════════════════════════════════════

def loco_cv(
    df:       pd.DataFrame,
    target:   str,
    features: list,
    model_fn,          # callable returning a fresh sklearn estimator
    model_name: str,
) -> dict:
    """
    Leave-One-Country-Out cross-validation.

    For each fold k:
      - Train on all countries except country k
      - Predict on country k
      - Record R² and RMSE for held-out country

    Returns dict with per-fold and aggregate metrics.

    Note: with G=8 folds, confidence intervals on CV metrics are wide.
    Report mean ± SD across folds rather than a single estimate.

    Note on SHAP under LOCO: SHAP is NOT computed per fold here.
    Per-fold SHAP on ≈25 held-out observations would be extremely noisy
    (variance dominates signal). SHAP is computed on the full-sample
    model instead, which is the standard practice for interpretability
    (Molnar 2022 §9.5). The LOCO loop serves exclusively for predictive
    performance evaluation.
    """
    feats_a   = [f for f in features if f in df.columns]
    work_cols = [target, ENTITY_COL] + feats_a
    df_work   = df[work_cols].dropna()

    fold_results = []
    for country in COUNTRIES:
        train = df_work[df_work[ENTITY_COL] != country]
        test  = df_work[df_work[ENTITY_COL] == country]

        if len(train) < 20 or len(test) < 3:
            fold_results.append({"country": country, "r2": np.nan, "rmse": np.nan})
            continue

        X_train = train[feats_a].values
        y_train = train[target].values
        X_test  = test[feats_a].values
        y_test  = test[target].values

        m = model_fn()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_results.append({"country": country, "r2": r2, "rmse": rmse})

    r2s   = [r["r2"]   for r in fold_results if not np.isnan(r["r2"])]
    rmses = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]

    return {
        "model":    model_name,
        "folds":    fold_results,
        "r2_mean":  float(np.mean(r2s))  if r2s else np.nan,
        "r2_sd":    float(np.std(r2s))   if r2s else np.nan,
        "rmse_mean":float(np.mean(rmses)) if rmses else np.nan,
        "rmse_sd":  float(np.std(rmses))  if rmses else np.nan,
        "n_valid_folds": len(r2s),
    }


# ════════════════════════════════════════════════════════════════════════
# CLUSTER BOOTSTRAP SHAP STABILITY
# ════════════════════════════════════════════════════════════════════════

def shap_cluster_bootstrap_ranks(
    df:        pd.DataFrame,
    target:    str,
    features:  list,
    model_fn,
    B:         int = 200,
    seed:      int = 42,
) -> dict:
    """
    Cluster bootstrap SHAP feature importance rankings.

    Because observations within each country are serially correlated,
    individual-row resampling violates the i.i.d. assumption of the
    ordinary bootstrap and underestimates the true variability of SHAP
    rankings (Cameron & Miller 2015).

    Correct approach: resample COUNTRIES with replacement, then retain
    all time periods for selected countries. This preserves within-country
    serial structure and reflects the actual unit of independent variation.

    Limitation: with G=8 clusters, there are only 8^8 ≈ 16M possible
    bootstrap samples. The bootstrap distribution is discrete and coarse.
    Rank SDs and hit-rate estimates should therefore be treated as
    approximate lower bounds on true uncertainty.

    For each of B bootstrap draws:
      1. Resample G countries with replacement.
      2. Concatenate all years for the selected (possibly repeated) countries.
         When a country appears k times, its rows are included k times.
      3. Fit the model on the bootstrap sample.
      4. Compute SHAP values on the ORIGINAL full data.
      5. Rank features by mean |SHAP|.

    Returns per-feature:
      - mean_rank       : mean rank across bootstrap draws (1 = most important)
      - sd_rank         : SD of rank across draws
      - prob_top3       : fraction of draws where feature ranked in top 3
      - prob_top5       : fraction of draws where feature ranked in top 5

    Reference:
      Molnar (2022), Interpretable Machine Learning, §9.6.
      Cameron & Miller (2015), JHR — cluster-robust bootstrap.
    """
    rng     = np.random.default_rng(seed)
    feats_a = [f for f in features if f in df.columns]
    df_work = df[[target, ENTITY_COL] + feats_a].dropna()

    # Full data used for SHAP evaluation (fixed, not resampled)
    X_all = df_work[feats_a].values
    y_all = df_work[target].values
    k     = len(feats_a)

    rank_matrix = []   # B × k matrix of ranks per draw

    for b in range(B):
        # Resample countries with replacement (cluster bootstrap)
        selected_countries = rng.choice(COUNTRIES, size=G, replace=True)
        frames = []
        for c in selected_countries:
            frames.append(df_work[df_work[ENTITY_COL] == c])
        boot_df = pd.concat(frames, ignore_index=True)

        X_b = boot_df[feats_a].values
        y_b = boot_df[target].values

        if len(X_b) < k + 2:
            # Degenerate sample (all same country, too few rows)
            continue

        m = model_fn()
        try:
            m.fit(X_b, y_b)
            exp      = shap.TreeExplainer(m)
            sv       = exp.shap_values(X_all)     # (n, k)
            mean_abs = np.abs(sv).mean(axis=0)    # (k,)
            # Rank: 1 = most important
            ranks    = stats.rankdata(-mean_abs)
            rank_matrix.append(ranks)
        except Exception:
            continue

    if not rank_matrix:
        return {}

    rank_mat = np.array(rank_matrix)   # (B_valid, k)
    B_valid  = rank_mat.shape[0]

    return {
        feat: {
            "mean_rank": float(rank_mat[:, i].mean()),
            "sd_rank":   float(rank_mat[:, i].std()),
            "prob_top3": float((rank_mat[:, i] <= 3).mean()),
            "prob_top5": float((rank_mat[:, i] <= 5).mean()),
        }
        for i, feat in enumerate(feats_a)
    }


# ════════════════════════════════════════════════════════════════════════
# ALE (ACCUMULATED LOCAL EFFECTS) PLOTS
# ════════════════════════════════════════════════════════════════════════

def compute_ale_1d(
    model,
    X:         np.ndarray,
    feat_idx:  int,
    n_bins:    int = 20,
) -> tuple:
    """
    Compute 1-D ALE for a single feature.

    ALE (Apley & Zhu 2020) is preferred over PDP when features are
    correlated, because it conditions the prediction on the local
    neighbourhood of each observation rather than marginalising over
    the full (possibly extrapolated) feature distribution.

    Algorithm:
      1. Divide feature into quantile-based bins.
      2. For each bin, compute the average change in prediction when
         the feature moves from the lower to upper bin edge, holding
         all other features at their observed values.
      3. Accumulate these local differences and centre.

    Returns:
      centres : bin centre values (length ≤ n_bins)
      ale     : accumulated local effect at each centre
    """
    x_feat = X[:, feat_idx]
    quantiles = np.unique(np.quantile(x_feat, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return np.array([]), np.array([])

    n_actual_bins = len(quantiles) - 1
    ale_local = np.zeros(n_actual_bins)
    counts    = np.zeros(n_actual_bins)

    for j in range(n_actual_bins):
        lo, hi = quantiles[j], quantiles[j + 1]
        # Observations whose feature value falls in this bin
        mask = (x_feat >= lo) & (x_feat < hi)
        if j == n_actual_bins - 1:
            mask = (x_feat >= lo) & (x_feat <= hi)
        if mask.sum() == 0:
            continue

        X_lo         = X[mask].copy();  X_lo[:, feat_idx] = lo
        X_hi         = X[mask].copy();  X_hi[:, feat_idx] = hi
        delta        = model.predict(X_hi) - model.predict(X_lo)
        ale_local[j] = delta.mean()
        counts[j]    = mask.sum()

    # Accumulate and centre
    ale_acc = np.cumsum(ale_local)
    ale_acc -= np.average(ale_acc, weights=np.maximum(counts, 1))

    centres = 0.5 * (quantiles[:-1] + quantiles[1:])
    return centres, ale_acc


# ════════════════════════════════════════════════════════════════════════
# WILCOXON COMPARISON HELPER
# ════════════════════════════════════════════════════════════════════════

def compare_rf_gb_loco(rf_cv: dict, gb_cv: dict) -> dict:
    """
    Wilcoxon signed-rank test on paired LOCO fold R² values.

    WARNING: with G=8 folds the test has very low power (the minimum
    achievable two-sided p-value with 8 non-zero differences is 0.0078).
    Results are reported for completeness but should be treated as
    purely descriptive, not as formal evidence for one model over the other.
    """
    rf_map = {r["country"]: r["r2"] for r in rf_cv["folds"]}
    gb_map = {r["country"]: r["r2"] for r in gb_cv["folds"]}

    # Only countries where both models produced valid R²
    common = [c for c in COUNTRIES
              if not np.isnan(rf_map.get(c, np.nan))
              and not np.isnan(gb_map.get(c, np.nan))]
    rf_vals = [rf_map[c] for c in common]
    gb_vals = [gb_map[c] for c in common]

    if len(common) < 4:
        return {"n_pairs": len(common), "note": "Too few pairs for reliable test."}

    stat, pval = stats.wilcoxon(rf_vals, gb_vals, alternative="two-sided",
                                zero_method="wilcox")
    return {
        "n_pairs":   len(common),
        "rf_r2_mean": float(np.mean(rf_vals)),
        "gb_r2_mean": float(np.mean(gb_vals)),
        "w_stat":    float(stat),
        "p_value":   float(pval),
        "note": (f"G={len(common)} pairs → very low power. "
                 "Treat p-value as descriptive only."),
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN LOOP OVER TASKS
# ════════════════════════════════════════════════════════════════════════
section("06-B — ML ESTIMATION (LOCO-CV, SHAP, ALE)")

ml_results      = {}
all_shap_values = {}
fitted_models   = {}

RF_KWARGS = dict(
    n_estimators=500,
    max_depth=4,          # Conservative: prevents overfitting with N≈175-200 obs total.
    min_samples_leaf=4,   # Conservative: avoids splitting on country-specific noise.
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,       # Retained as internal diagnostic only (not reported as CV metric).
    random_state=SEED,
)
GB_KWARGS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=4,
    random_state=SEED,
)

for task in TASKS:
    target      = task["target"]
    features    = task["features"]
    task_name   = task["name"]
    key_feature = task["key_feature"]
    feats_a = [f for f in features if f in df.columns]

# eliminar columnas completamente vacías o constantes
    feats_a = [
        f for f in feats_a
        if df[f].notna().sum() > 10 and df[f].nunique() > 1
    ]

    subsection(task_name)
    df_check = df[[target] + feats_a]
    print("Missing summary:")
    print(df_check.isna().mean().sort_values(ascending=False))
    print("Rows before dropna:", len(df_check))
    print("Rows after dropna:", len(df_check.dropna()))

    # ── LOCO-CV for RF and GB ────────────────────────────────────────────
    rf_cv = loco_cv(df, target, feats_a,
                    lambda: RandomForestRegressor(**RF_KWARGS),
                    "Random Forest")
    gb_cv = loco_cv(df, target, feats_a,
                    lambda: GradientBoostingRegressor(**GB_KWARGS),
                    "Gradient Boosting")

    print(f"  {'Model':<22} {'R² mean':>10} {'R² SD':>9} {'RMSE mean':>12} {'Valid folds':>12}")
    print("  " + "─" * 68)
    for cv in [rf_cv, gb_cv]:
        print(f"  {cv['model']:<22} {cv['r2_mean']:>10.4f} {cv['r2_sd']:>9.4f} "
              f"{cv['rmse_mean']:>12.4f} {cv['n_valid_folds']:>12}")
    print(warn(f"  Note: G={G} LOCO folds → R² estimates have wide uncertainty. "
               "SD across folds is as informative as the mean."))

    # ── RF vs GB Wilcoxon comparison ─────────────────────────────────────
    wil = compare_rf_gb_loco(rf_cv, gb_cv)
    print(f"\n  RF vs GB Wilcoxon signed-rank (paired LOCO R²):")
    print(f"    N pairs: {wil['n_pairs']}  |  W={wil.get('w_stat', 'n/a')}  "
          f"p={wil.get('p_value', 'n/a')}")
    print(warn(f"    {wil['note']}"))

    # ── Fit RF on full data for SHAP / ALE / permutation importance ──────

    df_work = df[[target] + feats_a + [ENTITY_COL]]

# separar X e y sin dropna agresivo
    mask = ~df_work[target].isna()

    X = df_work.loc[mask, feats_a]
    y = df_work.loc[mask, target]

    # imputación SOLO en X
    imp = SimpleImputer(strategy="median")
    X_full = imp.fit_transform(X)

# FIX CRÍTICO: reconciliar columnas reales vs imputadas
    if X_full.shape[1] != len(feats_a):
        print(err(
            f"Feature mismatch: X_full has {X_full.shape[1]} columns "
            f"but feats_a has {len(feats_a)}"
        ))

    # reconstruir nombres reales desde dataframe original
    feats_a = list(X.columns)

    y_full = y.values
    if X_full.shape[0] == 0 or y_full.shape[0] == 0:
        print(err(f"No usable data for {task_name} after preprocessing"))
        ml_results[task_name] = {
            "error": "empty_after_imputation"
        }
        continue
    if X_full.shape[0] < 30:
        print(warn(f"Skipping {task_name}: insufficient usable cases ({len(X_full)})"))
        ml_results[task_name] = {
            "error": "insufficient_data",
            "n_complete_cases": len(X_full)
        }
        continue

    rf_full = RandomForestRegressor(**RF_KWARGS)
    rf_full.fit(X_full, y_full)
    fitted_models[task_name] = {"model": rf_full, "features": feats_a, "data": df_work}

    # OOB R² is kept as an internal diagnostic to assist hyperparameter
    # assessment, but is NOT used as a performance claim (it is in-sample
    # and will be optimistic relative to LOCO). It is not exported.
    oob_r2 = rf_full.oob_score_
    print(f"\n  RF OOB R² (internal diagnostic — in-sample, optimistic): {oob_r2:.4f}")
    print(warn("  OOB R² is a training-data statistic, NOT a LOCO out-of-country R². "
               "Do not cite OOB R² as evidence of predictive performance."))

    # ── Gini importance ──────────────────────────────────────────────────
    gini_imp = pd.Series(rf_full.feature_importances_, index=feats_a).sort_values(ascending=False)

    # ── Permutation importance (full-sample model, all data)
    # Permutation importance is an interpretability diagnostic for the
    # fitted model, not a predictive metric (Molnar 2022 §8.5). Computing
    # it on the full sample is standard practice. Computing it per LOCO
    # fold on ≈25 held-out observations would introduce far more variance
    # than it would reduce bias, making the ranking less reliable.
    perm_imp = permutation_importance(
        rf_full, X_full, y_full, n_repeats=100, random_state=SEED
    )
    perm_s = pd.Series(perm_imp.importances_mean, index=feats_a).sort_values(ascending=False)
    perm_d = pd.Series(perm_imp.importances_std,  index=feats_a)

    # ── SHAP values (full-sample model) ──────────────────────────────────
    # SHAP is computed on the full-sample model following standard practice.
    # Limitation: SHAP values reflect in-sample feature associations of
    # a model trained on G-1 countries; they may not extrapolate to the
    # held-out country. This is an inherent limitation of explanatory ML
    # with small G and is noted in the methodological caveats.
    explainer  = shap.TreeExplainer(rf_full)
    shap_vals  = explainer.shap_values(X_full)
    shap_mean  = pd.Series(np.abs(shap_vals).mean(axis=0),
                           index=feats_a).sort_values(ascending=False)
    all_shap_values[task_name] = {
        "shap_matrix": shap_vals,
        "X":           X_full,
        "features":    feats_a,
        "entity":      df_work[ENTITY_COL].values,
    }

    # ── SHAP cluster bootstrap stability ─────────────────────────────────
    print(f"\n  Computing SHAP cluster bootstrap stability (B=200, cluster=country)...")
    shap_rank_stability = shap_cluster_bootstrap_ranks(
        df, target, feats_a,
        lambda: RandomForestRegressor(**RF_KWARGS),
        B=200, seed=SEED,
    )

    # ── Feature importance table ─────────────────────────────────────────
    print(f"\n  {'Feature':<35} {'Gini':>8} {'Perm±SD':>14} {'|SHAP|':>9} "
          f"{'SHAP rank':>10} {'Rank SD':>8} {'P(top3)':>8} {'P(top5)':>8}")
    print("  " + "─" * 105)

    for feat in shap_mean.index:
        label    = FEATURE_LABELS.get(feat, feat)[:33]
        gini     = gini_imp.get(feat, np.nan)
        perm     = perm_s.get(feat, np.nan)
        perm_sd  = perm_d.get(feat, np.nan)
        sv_mean  = shap_mean.get(feat, np.nan)
        stab     = shap_rank_stability.get(feat, {})
        mean_rk  = stab.get("mean_rank", np.nan)
        sd_rk    = stab.get("sd_rank",   np.nan)
        pt3      = stab.get("prob_top3",  np.nan)
        pt5      = stab.get("prob_top5",  np.nan)
        from utils import BOLD, RESET
        print(
            f"  {BOLD if feat==key_feature else ''}{label:<35}{RESET} "
            f"{gini:>8.4f} "
            f"{perm:>7.4f}±{perm_sd:<5.4f} "
            f"{sv_mean:>9.4f} "
            f"{mean_rk:>10.2f} "
            f"{sd_rk:>8.2f} "
            f"{pt3:>8.2f} "
            f"{pt5:>8.2f}"
        )

    # Key feature rank
    key_perm_rank = list(perm_s.index).index(key_feature) + 1 if key_feature in perm_s.index else None
    key_shap_rank = list(shap_mean.index).index(key_feature) + 1 if key_feature in shap_mean.index else None

    if key_perm_rank and key_shap_rank:
        print(f"\n  Key feature '{key_feature}':")
        print(f"    Permutation rank: #{key_perm_rank}/{len(feats_a)}")
        print(f"    SHAP rank:        #{key_shap_rank}/{len(feats_a)}")
        if key_perm_rank <= 3 and key_shap_rank <= 3:
            print(ok(f"  Key feature ranks in top 3 by both importance metrics."))
        else:
            print(warn(f"  Key feature does not rank in top 3. Inspect collinear features."))

    ml_results[task_name] = {
        "rf_cv": rf_cv, "gb_cv": gb_cv,
        "wilcoxon_rf_gb": wil,
        "oob_r2_internal": float(oob_r2),  # internal diagnostic only, not exported
        "gini_top5":       gini_imp.head(5).to_dict(),
        "perm_top5":       perm_s.head(5).to_dict(),
        "shap_top5":       shap_mean.head(5).to_dict(),
        "key_perm_rank":   key_perm_rank,
        "key_shap_rank":   key_shap_rank,
        "shap_rank_stab":  shap_rank_stability,
    }

# ════════════════════════════════════════════════════════════════════════
# CONVERGENCE TABLE: FE vs ML
# ════════════════════════════════════════════════════════════════════════
section("06-C — CONVERGENCE TABLE: FE vs ML")

print("""
  Convergence principle (Athey & Imbens 2019):
  If the variables that matter causally (significant in FE) also rank
  highly in ML importance, this triangulation strengthens the narrative.
  Divergence signals potential non-linearities or confounders.
""")

fe_json = DIRS["json"].parent / "json" / "02_fe_results.json"
try:
    with open(fe_json) as f: fe_data = json.load(f)
except Exception:
    fe_data = {}

# Column header
print(f"  {'Task':<35} {'FE coef':>10} {'FE p-val':>10} {'FE sig':>8} "
      f"{'Direction':>10} {'Perm rank':>10} {'SHAP rank':>10} {'Convergence':>14}")
print("  " + "─" * 112)

for task in TASKS:
    task_name   = task["name"]
    key_feature = task["key_feature"]
    ml_res      = ml_results.get(task_name, {})
    perm_rk     = ml_res.get("key_perm_rank")
    shap_rk     = ml_res.get("key_shap_rank")

    # FE results for this equation's key feature
    eq_map = {"inst_avg": "EQ1", "fdi_percent_gdp": "EQ2", "gdp_growth": "EQ3"}
    eq_key = eq_map.get(task["target"], "")
    fe_eq  = fe_data.get(eq_key, {})

    fe_coef  = fe_eq.get("coef_key",     np.nan)
    fe_pval  = fe_eq.get("pval_key_cl",  np.nan)
    fe_sig   = sig_stars(fe_pval) if not np.isnan(float(fe_pval if fe_pval is not None else np.nan)) else "?"
    fe_coef_str = f"{fe_coef:.4f}" if fe_coef is not None and not np.isnan(float(fe_coef)) else "?"
    fe_pval_str = f"{fe_pval:.4f}" if fe_pval is not None and not np.isnan(float(fe_pval)) else "?"

    # Effect direction from FE coefficient
    direction = ""
    if fe_coef is not None and not np.isnan(float(fe_coef)):
        direction = "Neg" if float(fe_coef) < 0 else "Pos"

    converge = ""
    if perm_rk and shap_rk:
        if perm_rk <= 3 and shap_rk <= 3:
            converge = ok("Strong")
        elif perm_rk <= 5 or shap_rk <= 5:
            converge = warn("Partial")
        else:
            converge = err("Weak")

    print(f"  {task_name:<35} {fe_coef_str:>10} {fe_pval_str:>10} {fe_sig:>8} "
          f"{direction:>10} {str(perm_rk):>10} {str(shap_rk):>10} {converge:>14}")

print("""
  Notes:
  - FE coef and p-value are cluster-robust estimates from Module 02.
  - FE sig: * p<0.10  ** p<0.05  *** p<0.01.
  - Direction: sign of the FE coefficient for the key feature.
  - Perm/SHAP rank: rank among all features in the ML model (1 = most important).
  - Convergence: Strong = key feature ranks top 3 by both metrics.
""")

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
section("06-D — FIGURES")

# ── Figure A: LOCO-CV performance ────────────────────────────────────────
fig_cv, axes_cv = plt.subplots(1, len(TASKS), figsize=(14, 5))
fig_cv.suptitle(
    "Leave-One-Country-Out Cross-Validation (LOCO-CV)\n"
    "Random Forest & Gradient Boosting  |  Note: G=8 folds → wide uncertainty",
    fontsize=11, fontweight="bold"
)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, task in zip(axes_cv, TASKS):
    task_name = task["name"]
    ml_res    = ml_results[task_name]
    rf_folds  = {r["country"]: r["r2"] for r in ml_res["rf_cv"]["folds"]}
    gb_folds  = {r["country"]: r["r2"] for r in ml_res["gb_cv"]["folds"]}

    # Use only countries where both models produced valid R²
    valid_countries = [
        c for c in COUNTRIES
        if not np.isnan(rf_folds.get(c, np.nan))
        and not np.isnan(gb_folds.get(c, np.nan))
    ]
    rf_r2s = [rf_folds[c] for c in valid_countries]
    gb_r2s = [gb_folds[c] for c in valid_countries]

    x = np.arange(len(valid_countries))
    w = 0.35
    ax.bar(x - w/2, rf_r2s, w, label="Random Forest",
           color=COLORS[0], alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, gb_r2s, w, label="Gradient Boosting",
           color=COLORS[1], alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_countries, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Out-of-country R²", fontsize=9)
    ax.set_title(f"{task_name}\n(held-out country)", fontsize=9)
    ax.legend(fontsize=8)

fig_cv.tight_layout()
fig_cv.savefig(DIRS["figures"] / "08_ml_loco_cv.png", dpi=300, bbox_inches="tight")
plt.close(fig_cv)
print(ok("Figure saved → figures/08_ml_loco_cv.png"))

# ── Figure B: Feature importance comparison (all tasks) ──────────────────
fig_fi, axes_fi = plt.subplots(len(TASKS), 3, figsize=(15, 5 * len(TASKS)))
fig_fi.suptitle(
    "Feature Importance — Gini, Permutation, and SHAP\n"
    "(Exploratory — not causal evidence)",
    fontsize=12, fontweight="bold"
)

for row, task in enumerate(TASKS):
    task_name   = task["name"]
    key_feature = task["key_feature"]
    feats_a     = fitted_models[task_name]["features"]
    ml_res      = ml_results[task_name]

    rf_full = fitted_models[task_name]["model"]
    df_work = fitted_models[task_name]["data"]
    X_full  = df_work[feats_a].values

    gini_s = pd.Series(rf_full.feature_importances_,
                       index=feats_a).sort_values(ascending=False).head(8)
    perm_imp2 = permutation_importance(
        rf_full, X_full, df_work[task["target"]].values,
        n_repeats=50, random_state=SEED
    )
    perm_s2 = pd.Series(perm_imp2.importances_mean,
                        index=feats_a).sort_values(ascending=False).head(8)
    perm_sd2 = pd.Series(perm_imp2.importances_std, index=feats_a)

    # SHAP from cached values
    sv_info    = all_shap_values[task_name]
    shap_mean2 = pd.Series(
        np.abs(sv_info["shap_matrix"]).mean(axis=0),
        index=sv_info["features"]
    ).sort_values(ascending=False).head(8)

    def bar_color_list(index, key_feat):
        return [COLORS[1] if f == key_feat else COLORS[0] for f in index]

    # Gini
    ax = axes_fi[row, 0]
    y_g = np.arange(len(gini_s))[::-1]
    ax.barh(y_g, gini_s.values,
            color=bar_color_list(gini_s.index, key_feature), alpha=0.8, height=0.6)
    ax.set_yticks(y_g)
    ax.set_yticklabels(
        [FEATURE_LABELS.get(f, f)[:28] for f in gini_s.index], fontsize=8
    )
    ax.set_xlabel("Gini Importance", fontsize=9)
    ax.set_title(f"{task_name}\nGini Importance", fontsize=9)

    # Permutation ± SD
    ax = axes_fi[row, 1]
    y_p = np.arange(len(perm_s2))[::-1]
    ax.barh(y_p, perm_s2.values,
            xerr=perm_sd2[perm_s2.index].values,
            color=bar_color_list(perm_s2.index, key_feature), alpha=0.8,
            height=0.6, capsize=3, error_kw={"linewidth": 1})
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_p)
    ax.set_yticklabels(
        [FEATURE_LABELS.get(f, f)[:28] for f in perm_s2.index], fontsize=8
    )
    ax.set_xlabel("Permutation Importance ± SD", fontsize=9)
    ax.set_title("Permutation Importance\n(± 1 SD)", fontsize=9)

    # SHAP
    ax = axes_fi[row, 2]
    y_s = np.arange(len(shap_mean2))[::-1]
    ax.barh(y_s, shap_mean2.values,
            color=bar_color_list(shap_mean2.index, key_feature), alpha=0.8, height=0.6)
    ax.set_yticks(y_s)
    ax.set_yticklabels(
        [FEATURE_LABELS.get(f, f)[:28] for f in shap_mean2.index], fontsize=8
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("SHAP Feature Importance\n(TreeExplainer)", fontsize=9)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS[1], label="Key causal feature"),
    Patch(facecolor=COLORS[0], label="Other features"),
]
fig_fi.legend(handles=legend_elements, loc="upper right", fontsize=9)
fig_fi.tight_layout(rect=[0, 0, 0.97, 1])
fig_fi.savefig(DIRS["figures"] / "09_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close(fig_fi)
print(ok("Figure saved → figures/09_feature_importance.png"))

# ── Figure C: SHAP scatter + ALE for key variables ───────────────────────
# ALE is used instead of PDP because the predictors in this panel model
# are economically correlated (GDP, institutions, FDI, violence all move
# together). PDP averages predictions over the marginal distribution of
# all other features, which can produce extrapolation artefacts when
# features are correlated. ALE conditions on the local neighbourhood
# instead (Apley & Zhu 2020), making it more appropriate here.
fig_shap, axes_shap = plt.subplots(2, len(TASKS), figsize=(15, 9))
fig_shap.suptitle(
    "SHAP Dependence Plots & ALE Plots — Key Causal Variables\n"
    "(Exploratory — shows non-linearity, not causal effect)\n"
    "ALE used instead of PDP due to correlated predictors (Apley & Zhu 2020)",
    fontsize=10, fontweight="bold"
)

for col, task in enumerate(TASKS):
    task_name   = task["name"]
    key_feature = task["key_feature"]
    feats_a     = fitted_models[task_name]["features"]
    rf_full     = fitted_models[task_name]["model"]
    df_work     = fitted_models[task_name]["data"]
    X_full      = df_work[feats_a].values
    entity_arr  = df_work[ENTITY_COL].values

    if key_feature not in feats_a:
        continue
    feat_idx = feats_a.index(key_feature)

    sv_info  = all_shap_values[task_name]
    sv_key   = sv_info["shap_matrix"][:, feat_idx]
    x_key    = X_full[:, feat_idx]

    # SHAP scatter (row 0)
    ax_top = axes_shap[0, col]
    for i, country in enumerate(COUNTRIES):
        mask = entity_arr == country
        ax_top.scatter(x_key[mask], sv_key[mask],
                       color=COLORS[i % len(COLORS)], s=18, alpha=0.7,
                       label=country if col == 0 else None)
    ax_top.axhline(0, color="black", linewidth=0.7)
    ax_top.set_xlabel(FEATURE_LABELS.get(key_feature, key_feature), fontsize=9)
    ax_top.set_ylabel("SHAP value", fontsize=9)
    ax_top.set_title(f"{task_name[:30]}\nSHAP dependence: {key_feature[:20]}", fontsize=8)
    if col == 0:
        ax_top.legend(fontsize=6, ncol=2)

    # ALE plot (row 1)
    ax_bot = axes_shap[1, col]
    try:
        ale_x, ale_y = compute_ale_1d(rf_full, X_full, feat_idx, n_bins=20)
        if len(ale_x) > 1:
            ax_bot.plot(ale_x, ale_y, color=COLORS[col], linewidth=2)
            ax_bot.axhline(0, color="grey", linestyle="--", linewidth=0.8,
                           alpha=0.7, label="ALE = 0 (no local effect)")
            ax_bot.set_xlabel(FEATURE_LABELS.get(key_feature, key_feature), fontsize=9)
            ax_bot.set_ylabel(f"ALE on {task['target']}", fontsize=9)
            ax_bot.set_title(
                "Accumulated Local Effects (ALE)\n"
                "(centred; conditions on local neighbourhood)", fontsize=8
            )
            ax_bot.legend(fontsize=8)
        else:
            ax_bot.text(0.5, 0.5, "Insufficient bins for ALE",
                        ha="center", va="center", transform=ax_bot.transAxes, fontsize=8)
    except Exception as e:
        ax_bot.text(0.5, 0.5, f"ALE failed:\n{e}", ha="center", va="center",
                    transform=ax_bot.transAxes, fontsize=8)

fig_shap.tight_layout()
fig_shap.savefig(DIRS["figures"] / "10_shap_ale.png", dpi=300, bbox_inches="tight")
plt.close(fig_shap)
print(ok("Figure saved → figures/10_shap_ale.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("06-E — EXPORT")

export = {}
for task_name, res in ml_results.items():
    export[task_name] = {
        "rf_cv_r2_mean":   res["rf_cv"]["r2_mean"],
        "rf_cv_r2_sd":     res["rf_cv"]["r2_sd"],
        "gb_cv_r2_mean":   res["gb_cv"]["r2_mean"],
        "gb_cv_r2_sd":     res["gb_cv"]["r2_sd"],
        # OOB R² intentionally excluded from export: it is an
        # internal diagnostic only and must not appear in the paper.
        "wilcoxon_rf_gb":  res["wilcoxon_rf_gb"],
        "gini_top5":       res["gini_top5"],
        "perm_top5":       res["perm_top5"],
        "shap_top5":       res["shap_top5"],
        "key_perm_rank":   res["key_perm_rank"],
        "key_shap_rank":   res["key_shap_rank"],
        "shap_rank_stability": {
            k: {
                "mean_rank": v["mean_rank"],
                "sd_rank":   v["sd_rank"],
                "prob_top3": v["prob_top3"],
                "prob_top5": v["prob_top5"],
            }
            for k, v in res["shap_rank_stab"].items()
        },
    }

save_json(export, DIRS["json"] / "06_ml_results.json")
print(f"\n{bold('Module 06 complete. Pipeline finished.')}")

print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  IMPORTANT REMINDER                                             ║
  ║  ML results (SHAP, permutation importance, LOCO R²)             ║
  ║  are EXPLORATORY and NON-CAUSAL.                                ║
  ║  They triangulate — they do not identify causal effects.        ║
  ║  Primary causal evidence: Module 02 (Two-Way FE) and            ║
  ║  Module 04 (Wild Cluster Bootstrap inference).                  ║
  ║                                                                 ║
  ║  SHAP limitation: computed on the full-sample model.            ║
  ║  Values reflect in-sample feature associations and may not      ║
  ║  generalise to held-out countries.                              ║
  ║                                                                 ║
  ║  Bootstrap stability limitation: G=8 clusters → discrete        ║
  ║  bootstrap distribution. Treat rank SDs and hit rates as        ║
  ║  approximate lower bounds on true uncertainty.                  ║
  ╚══════════════════════════════════════════════════════════════════╝
""")