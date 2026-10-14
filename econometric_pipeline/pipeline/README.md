# Econometric Pipeline: Violence → Institutions → FDI → Growth

**18 Latin American and Caribbean countries | 2000–2024**

---

## Overview

This pipeline implements the empirical strategy for a paper studying the chain through
which violence is hypothesised to affect economic growth. All estimates are associations
under two-way fixed effects, not causal effects; the chain is not robustly significant on
the 18-country panel (see the repository README):

```
Violence (homicide rate)
       ↓  [Equation 1]
Institutional Quality (WGI: 6 governance dimensions, incl. Rule of Law, Control of Corruption, Political Stability)
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
# from the repository root
pip install -r requirements.txt          # pinned versions used for the published results
pip install -r requirements-dev.txt      # adds pytest

cd econometric_pipeline/pipeline
python run_pipeline.py                   # all 15 modules, then the PDF report
python run_pipeline.py --skip 06 08      # skip modules; the report reuses their stored JSON outputs
python run_pipeline.py --data /path/to/panel.csv   # default: panel_ready_for_modeling.csv in this folder

cd ../..
python -m pytest                         # tests (output consistency, data integrity, documentation vs JSON)
```

Modules 04 (bootstrap), 08 and 15 (MCMC, about 15-25 minutes each) dominate the run time. A fixed
seed makes every module reproducible: a clean re-run of all 15 modules reproduces the stored JSON
outputs exactly.

---

## Module Architecture

| Module | File | Purpose |
|--------|------|---------|
| 01 | `01_data_preparation.py` | Data validation, institution index (z-score average and PCA), stationarity, summary statistics |
| 02 | `02_panel_estimation.py` | Two-Way FE estimation of EQ1-EQ3 (primary), pooled OLS benchmark |
| 03 | `03_diagnostics.py` | Wooldridge, Pesaran CD, Modified Wald, VIF, DFBETA |
| 04 | `04_bootstrap_inference.py` | Wild Cluster Bootstrap (Webb weights, B=999), CR2 SE, cluster bootstrap of the indirect effect |
| 05 | `05_robustness.py` | LOCO, time windows, alternative lags / indices / controls, placebo, panel composition (11 vs 18 countries) |
| 06 | `06_ml_triangulation.py` | RF + GB (LOCO-CV), SHAP, ALE, convergence table |
| 07 | `07_cointegration.py` | Panel unit roots (Fisher-ADF), Engle-Granger / Kao cointegration, error-correction model |
| 08 | `08_growth_ceiling_risk.py` | Hierarchical Bayesian quantile regression for the growth ceiling (PyMC / NUTS) |
| 09 | `09_synthetic_control.py` | Synthetic control for El Salvador (placebo-in-space, robustness checks) |
| 10 | `10_arch_lm_test.py` | Engle ARCH-LM test per country, Fisher-combined |
| 11 | `11_spec_curve.py` | Specification curve over all 63 definitions of the institution index |
| 12 | `12_spatial_diagnostics.py` | Cross-country dependence: Pesaran CD, permutation Moran's I, SLX spillovers, donor contamination |
| 13 | `13_causal_link_scan.py` | FDR-controlled scan of all lagged links among nine variables, with placebo calibration |
| 14 | `14_wgi_measurement.py` | WGI measurement error from the official 90% intervals: noise, distinguishable changes, reliability, EQ1 by dimension group |
| 15 | `15_growth_ceiling_robustness.py` | Growth ceiling: leave-one-country-out and Bayesian fit without El Salvador |
| 16 | `16_link_scan_lags.py` | Optional extension, run on its own (not in `run_pipeline.py`): the Module 13 scan with the cause lagged 1, 2 and 3 years (216 tests, one multiplicity family); null result |
| — | `utils.py` | Shared utilities (all modules import from here) |
| — | `research_report.py` | PDF report builder |
| — | `run_pipeline.py` | Master runner |

The geometric-deep-learning extension (global panel, baselines, graph neural networks) is separate:
see `gdl_extension/` at the repository root.

--------|------|---------|
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

### Governance (WGI governance scores, 0–100; 2025 revision: a linear rescaling of the estimate between two benchmark countries, not percentile ranks)
- `rule_of_law`
- `control_corruption`
- `political_stability`
- `voice_accountability`
- `government_effectiveness`
- `regulatory_quality`

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
- `remittances_percent_gdp` — tested as a robustness control variant only (Module 05); not in the primary control set (plausible mediator/collider, see Module 05 docstring)

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
  Rationale: with G=18 clusters, conventional clustered SE are severely
  undersized (Cameron & Miller 2015). Webb weights control size better
  than Rademacher weights for small G.
- **Secondary**: CR2 Bell-McCaffrey bias-corrected clustered SE.
- **Reported alongside**: Driscoll-Kraay HAC (robust to cross-sectional
  dependence detected by Pesaran CD test).

### Institution index
- **Primary**: standardised average of all six z-scored WGI dimensions
  (rule of law, control of corruption, political stability, voice &
  accountability, government effectiveness, regulatory quality).
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

3. **Arellano-Bond GMM**: With N=18 and T≈24, full AB-GMM would suffer severe
   instrument proliferation. Module 04 explicitly states this and does NOT
   fake AB-GMM with first-difference OLS.

4. **Wooldridge test**: Python has no exact equivalent of Stata's `xtserial`.
   The implementation follows Wooldridge (2002) p.282-283 manually and
   flags this limitation explicitly.

5. **N=18 clusters**: All inference warnings about small cluster count are
   printed prominently. Wild bootstrap is the recommended inference procedure.

6. **Perception-based institutions**: the WGI are aggregates of expert and survey perceptions
   whose own margin of error is of the order of the within-country variation used by the
   fixed effects (Module 14); Rule of Law and Political Stability draw on crime and violence items.

7. **Cross-country dependence**: with 18 countries the spatial diagnostics (Module 12) have
   little power; EQ3 residuals show regional dependence, EQ1 and EQ2 residuals do not.

---

## Outputs

```
figures/   22 PNG files numbered by module
             01-03 panel overview, institution indices, correlations | 04-05 FE residuals, diagnostics
             06, 06b bootstrap and mediation | 07 robustness | 08-10 ML (LOCO-CV, importance, SHAP/ALE)
             11 cointegration | 12-13 growth ceiling (Bayesian, scenario) | 14 Bukele paradox
             15, 15b synthetic control | 16 specification curve | 17 spatial dependence
             18 causal-link map | 19 WGI measurement | 20 growth-ceiling leave-one-out

tables/    summary_statistics.tex, se_comparison.tex, growth_ceiling_bayesian.tex

json/      one file per module: 01_metadata.json ... 15_growth_ceiling_robustness.json
           (04 also writes 04_mediation.json). These are the single source of every number in the
           READMEs, the PDF report and the manuscript; tests/ checks that they agree.

econometric_report.pdf   full technical report (Spanish), regenerated by run_pipeline.py
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
- Abadie, Diamond & Hainmueller (2010); Abadie (2021) — Synthetic control, *JASA*; *Journal of Economic Literature*
- Adrian, Boyarchenko & Giannone (2019) — Vulnerable growth, *American Economic Review*
- Benjamini & Hochberg (1995) — False discovery rate, *Journal of the Royal Statistical Society B*
- Engle (1982) — ARCH, *Econometrica*; Engle & Granger (1987); Kao (1999); Maddala & Wu (1999) — cointegration and panel unit roots
- Jiang, Jain, Romano & Boyce (2025); Kaufmann & Kraay (2024) — WGI methodology, World Bank
- Moran (1950) — Spatial autocorrelation, *Biometrika*
- Simonsohn, Simmons & Nelson (2020) — Specification curve analysis, *Nature Human Behaviour*
- Yu & Moyeed (2001) — Bayesian quantile regression, *Statistics & Probability Letters*
- Wooldridge (2002) — *Econometric Analysis of Cross Section and Panel Data*
