# Econometric Pipeline: Violence → Institutions → FDI → Growth

**Central America, Colombia, Dominican Republic | 2000–2024**

---

## Overview

This pipeline implements the empirical strategy for a paper studying the causal
mechanism through which violence affects economic growth:

```
Violence (homicide rate)
       ↓  [Equation 1]
Institutional Quality (WGI: Rule of Law, Control of Corruption, Political Stability)
       ↓  [Equation 2]
Foreign Direct Investment (FDI % GDP)
       ↓  [Equation 3]
GDP Growth
```

All code is designed to be methodologically defensible at the Q1 journal level
(Journal of Development Economics, World Development, Journal of Econometrics).

---

## Quick Start

```bash
# Install dependencies
pip install linearmodels statsmodels scikit-learn shap pandas numpy \
            matplotlib scipy arch joblib

# Place your CSV in this directory as:
#   panel_ready_for_modeling.csv
# OR specify the path:

python run_pipeline.py --data /path/to/your/panel.csv
```

---

## Module Architecture

| Module | File | Purpose |
|--------|------|---------|
| 01 | `01_data_preparation.py` | Data validation, institution index, stationarity, summary stats |
| 02 | `02_panel_estimation.py` | Two-Way FE estimation (primary), pooled OLS benchmark |
| 03 | `03_diagnostics.py` | Wooldridge, Pesaran CD, Modified Wald, VIF, DFBETA |
| 04 | `04_bootstrap_inference.py` | Wild Cluster Bootstrap (Webb weights, B=999) + CR2 SE |
| 05 | `05_robustness.py` | LOCO, time windows, alt. lags, alt. indices, placebo |
| 06 | `06_ml_triangulation.py` | RF + GB (LOCO-CV), SHAP, PDP, convergence table |
| — | `utils.py` | Shared utilities (all modules import from here) |
| — | `run_pipeline.py` | Master runner |

---

## Data Requirements

The CSV must contain the following columns:

### Panel identifiers
- `country_code` — ISO 3-letter country code
- `year` — integer year

### Violence variable
- `homicide_rate_log` — log(homicide rate per 100,000)
- `homicide_rate_log_lag1`, `homicide_rate_log_lag2`, `homicide_rate_log_lag3`
  — **pre-computed** lags (must be in dataset; pipeline does NOT recompute)

### Governance (WGI percentile ranks, 0–100)
- `rule_of_law`
- `control_corruption`
- `political_stability`

### Outcome and mediators
- `fdi_percent_gdp` — FDI as % of GDP
- `gdp_growth` — annual GDP growth rate (%)
- `gdp_per_capita_log` — log GDP per capita

### Controls
- `inflation`
- `unemployment`
- `exports_percent_gdp`
- `population_log`
- `tourist_arrivals_log`
- `trade_percent_gdp`

### Optional (for additional robustness)
- `time_trend` — integer time trend (Module 01 creates `year_c` from `year`)

---

## Methodological Choices

### Primary estimator
**Two-Way Fixed Effects** (`PanelOLS(entity_effects=True, time_effects=True)`)  
Absorbs: time-invariant country heterogeneity (geography, history, culture) +
common time shocks (global financial crisis, COVID-19).

### Standard errors
- **Primary**: Wild Cluster Bootstrap with Webb (2023) 6-point weights, B=999.
  Rationale: with G=8 clusters, conventional clustered SE are severely
  undersized (Cameron & Miller 2015). Webb weights control size better
  than Rademacher weights for small G.
- **Secondary**: CR2 Bell-McCaffrey bias-corrected clustered SE.
- **Reported alongside**: Driscoll-Kraay HAC (robust to cross-sectional
  dependence detected by Pesaran CD test).

### Institution index
- **Primary**: standardised average of three z-scored WGI dimensions.
  Equal weighting, transparent, easily replicated.
- **Robustness**: PCA-based index (PC1 of same z-scored variables).

### Lag structure
- Primary specification: `homicide_rate_log_lag1` (one-year lag).
- Robustness: contemporaneous, lag 2, lag 3 (using pre-computed columns).

---

## Key Methodological Limitations (explicitly flagged in output)

1. **Sequential OLS / Generated regressors**: The chain Eq1 → Eq2 → Eq3
   uses observed (not predicted) values in the primary spec to avoid the
   Pagan (1984) generated regressors problem. The bootstrap in Module 04
   partially addresses cross-equation uncertainty.

2. **IV/2SLS**: An IV architecture is prepared in Module 02 but requires
   external instruments (cocaine price index, rainfall anomaly) not present
   in the current dataset. Without valid instruments, endogeneity of violence
   cannot be fully resolved.

3. **Arellano-Bond GMM**: With N=8 and T≈24, full AB-GMM would suffer severe
   instrument proliferation. Module 04 explicitly states this and does NOT
   fake AB-GMM with first-difference OLS.

4. **Wooldridge test**: Python has no exact equivalent of Stata's `xtserial`.
   The implementation follows Wooldridge (2002) p.282-283 manually and
   flags this limitation explicitly.

5. **N=8 clusters**: All inference warnings about small cluster count are
   printed prominently. Wild bootstrap is the recommended inference procedure.

---

## Outputs

```
figures/
  01_panel_overview.png           Time series of key variables by country
  02_institution_indices.png      Primary vs PCA institution index scatter
  03_correlation_heatmap.png      Pooled correlation matrix
  04_fe_residuals.png             Residuals vs fitted for each equation
  05_diagnostics.png              Diagnostic test summary
  06_bootstrap_distributions.png  Wild bootstrap t-distributions
  07_robustness.png               Coefficient stability plots
  08_ml_loco_cv.png               LOCO cross-validation performance
  09_feature_importance.png       Gini, permutation, SHAP importance
  10_shap_pdp.png                 SHAP dependence + partial dependence plots

tables/
  summary_statistics.tex          LaTeX summary statistics table

json/
  01_metadata.json                Panel metadata, PCA info, KMO
  02_fe_results.json              FE coefficient estimates
  03_diagnostics.json             All diagnostic test statistics
  04_bootstrap.json               Bootstrap p-values, CIs, CR2 SE
  05_robustness.json              All robustness check results
  06_ml_results.json              ML importance rankings, CV metrics
```

---

## Literature References

- Angrist & Pischke (2009) — *Mostly Harmless Econometrics*
- Arellano & Bond (1991) — GMM for dynamic panels, *Review of Economic Studies*
- Bell & McCaffrey (2002) — CR2 SE, *Survey Methodology*
- Breiman (2001) — Random Forests, *Machine Learning*
- Cameron, Gelbach & Miller (2008) — Wild bootstrap, *Review of Economics and Statistics*
- Driscoll & Kraay (1998) — HAC for panels, *Review of Economics and Statistics*
- Greene (2000) — *Econometric Analysis*, 4th ed.
- Hausman (1978) — Specification tests, *Econometrica*
- Kaufmann et al. (2010) — WGI documentation, World Bank Policy Research WP
- Lundberg & Lee (2017) — SHAP, *NeurIPS*
- Mundlak (1978) — Pooling within/between, *Econometrica*
- Pagan (1984) — Generated regressors, *Review of Economic Studies*
- Pesaran (2004) — CD test, *Cambridge Working Papers in Economics*
- Webb (2023) — Wild bootstrap, *Canadian Journal of Economics*
- Wooldridge (2002) — *Econometric Analysis of Cross Section and Panel Data*
