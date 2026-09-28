# Violence, Institutions, and Growth in Latin America and the Caribbean

## Abstract

This repository implements an empirical research workflow for studying how violent crime affects institutional quality and, through that channel, economic growth. The project combines a structured panel-data econometric pipeline with machine-learning triangulation to examine whether the relationship between homicide exposure and growth is direct or mediated by institutional quality and foreign direct investment. The core analysis focuses on a panel of 11 countries — the original 8 in Central America, Colombia and the Dominican Republic, plus Mexico, Ecuador and Peru (added September 2026; see Panel Extension below) — over the period 2000–2024. The empirical strategy is designed to be transparent, reproducible, and suitable for academic review. The repository therefore emphasizes clear data preparation, robust inference, and documented estimation steps rather than ad hoc exploratory analysis.

## Research Question

How does violence, measured by homicide rates, affect institutional quality and subsequent economic performance, and is the relationship between crime and growth mediated by institutions and foreign direct investment?

## Methodology Overview

The project uses a longitudinal panel-data design with country and year effects. The baseline specification is a pooled OLS benchmark, followed by two-way fixed-effects models that absorb time-invariant country heterogeneity and common shocks. Inference is reported with conventional clustered standard errors and with more conservative robust alternatives, including Driscoll-Kraay standard errors to account for cross-sectional dependence, a CR2 (Bell-McCaffrey) bias-corrected estimator with Satterthwaite degrees of freedom, and wild cluster bootstrap inference for small-cluster panels. Because the three-equation chain (violence → institutions → FDI → growth) is estimated as separate single-equation models to avoid a generated-regressors problem, the hypothesized mechanism is additionally tested as a system: a country-cluster bootstrap of the product of the three path coefficients provides a formal test of the indirect effect, rather than relying on each link's significance in isolation. Robustness checks include leave-one-country-out re-estimation, alternative lag structures for every link in the chain (not only violence → institutions), a post-treatment ("bad control") check, country-specific linear trends, and a Fisher-ADF panel unit-root test. In addition, a principal component analysis is used to construct a summary institutional index from the World Governance Indicators, and machine-learning methods are used as an exploratory triangulation layer rather than as a substitute for causal estimation, with a leakage-free (fold-safe) reconstruction of the institution index under Leave-One-Country-Out cross-validation.

## Pipeline Architecture

The main entry point of the repository is the econometric pipeline under [econometric_pipeline/pipeline](econometric_pipeline/pipeline). This is the core reproducible system. All steps below are executed through the master runner in [econometric_pipeline/pipeline/run_pipeline.py](econometric_pipeline/pipeline/run_pipeline.py).

The workflow proceeds in eight stages:

1. Data loading: the pipeline reads a clean panel dataset containing country-year observations.
2. Cleaning and ETL: the data are validated, missingness is assessed, and institutional variables are standardized and combined into an index.
3. Feature engineering: the panel is enriched with time trends, lagged violence variables, and derived institutional measures.
4. PCA construction: a robustness institutional index is created using principal component analysis on the governance indicators.
5. Econometric estimation: two-way fixed-effects models estimate the violence-to-institutions, institutions-to-FDI, and FDI-to-growth relationships, each also re-estimated under a one-year-lag alternative.
6. Machine-learning triangulation: random forest and gradient boosting models are trained for exploratory comparison with the econometric estimates, using leave-one-country-out cross-validation with a fold-safe institution index.
7. Validation: diagnostics, bootstrap inference (including a cluster bootstrap of the full mediation chain's indirect effect), and robustness checks assess sensitivity to influential observations, lag choices, control-set choices, country-specific trends, non-stationarity, and model specification.
8. Panel cointegration & error-correction: for the pairs of variables found non-stationary in stage 7 (homicide rate, institutions, GDP per capita), a two-step Engle-Granger/Kao residual-based test checks whether they share a genuine long-run equilibrium before estimating an error-correction model; if not, this is reported as a finding in its own right rather than silently assumed away.

## Repository Structure

```text
data_analyzer/
├── econometric_pipeline/
│   └── pipeline/
│       ├── 01_data_preparation.py
│       ├── 02_panel_estimation.py
│       ├── 03_diagnostics.py
│       ├── 04_bootstrap_inference.py
│       ├── 05_robustness.py
│       ├── 06_ml_triangulation.py
│       ├── 07_cointegration.py
│       ├── run_pipeline.py
│       ├── utils.py
│       ├── research_report.py
│       ├── figures/
│       ├── json/
│       ├── tables/
│       └── README.md
├── archive/
│   ├── legacy_scripts/
│   └── legacy_data/
├── data_extraction/
├── documentation/
├── foundational_datasets/
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

### 2. Run the main pipeline

From the repository root:

```bash
python econometric_pipeline/pipeline/run_pipeline.py
```

Optional arguments include:

```bash
python econometric_pipeline/pipeline/run_pipeline.py --skip 4 5
python econometric_pipeline/pipeline/run_pipeline.py --data econometric_pipeline/pipeline/panel_ready_for_modeling.csv
```

### 3. Expected outputs

The pipeline generates:

- figures in [econometric_pipeline/pipeline/figures](econometric_pipeline/pipeline/figures), including the mediation bootstrap distribution (`06b_mediation_bootstrap.png`)
- tables in [econometric_pipeline/pipeline/tables](econometric_pipeline/pipeline/tables), including `se_comparison.tex` (standard-error sensitivity table, paste-ready for a manuscript)
- JSON summaries in [econometric_pipeline/pipeline/json](econometric_pipeline/pipeline/json), including `04_mediation.json` (indirect-effect bootstrap results)
- a final report PDF at [econometric_pipeline/pipeline/econometric_report.pdf](econometric_pipeline/pipeline/econometric_report.pdf)

## Data Sources

The empirical analysis draws on the following sources:

- World Bank World Development Indicators (WDI): macroeconomic and development indicators
- World Governance Indicators (WGI): rule of law, control of corruption, and political stability
- UNODC / crime statistics: homicide indicators
- Processed datasets used by the pipeline: the prepared panel input and the pipeline-generated intermediate files in [econometric_pipeline/pipeline](econometric_pipeline/pipeline)

## Data Provenance Fixes (September 2026)

A follow-up audit of the data-extraction chain (`data_extraction/`, `archive/legacy_scripts/`) — prompted by a hunch that mixed variable scales might be causing trouble — found the scales themselves were fine (mixing 0-100 percentiles, % GDP and logs in one regression doesn't bias FE/OLS or tree-based ML), but surfaced four real data bugs upstream of the pipeline described above:

1. **`homicide_rate` was silently winsorized at the 1st/99th percentile** before the log transform (`archive/legacy_scripts/build_master_dataset.py`). With 179 observations, the 4 clipped points were all El Salvador: its 2015/2016 violence peak (107.6 → 83.6, 85.1 → 83.6) and its 2023/2024 post-reform lows (2.24 → 5.43, 1.90 → 5.43) — exactly the abrupt security shift this project studies, flattened before it ever reached EQ1, and undocumented anywhere. Removed.
2. **`trade_percent_gdp` was 100% empty** across every observation, because it was computed as `exports_percent_gdp + imports_percent_gdp` but `imports_percent_gdp` was never extracted. This explains why "Extended controls" always failed in Module 05's robustness checks. Fixed by adding the missing World Bank indicator (`NE.IMP.GNFS.ZS`).
3. **`gdp_per_capita` mixed nominal (current US$) and real (constant 2015 US$) values for Guatemala specifically** — a gap-filling script pulled the wrong indicator for that one country, contaminating the within-country variation of a control used in every equation. Fixed to use the same indicator as the other seven countries.
4. **The only live WGI-extraction script pulled the wrong governance scale** (the "-2.5 to 2.5" estimate instead of the "0-100" score every module documents and expects) — dead code, since the pipeline actually sourced institutions from a manually-downloaded file that was never committed to the repo and whose exact vintage could not be verified. Fixed the indicator codes and pointed the merge step at the corrected live script instead, closing the reproducibility gap.

The panel is now balanced at the country-year level (200 rows = 8 × 25, up from 179 — the old chain dropped an entire row whenever *any* one source had a gap; individual variables still have their own genuine missing values, unchanged and still handled by `dropna()` downstream). Re-running the full pipeline on the corrected data leaves EQ1, EQ3 and the mediation test's conclusions unchanged, but **EQ2 (institutions → FDI) becomes substantially more robust**: p = 0.055 / 0.116 / 0.202 / 0.062 (clustered / Driscoll-Kraay / CR2 / wild bootstrap) under the old data → **p = 0.007 / 0.085 / 0.126 / 0.012** under the corrected data. See the commit range `0a750fe`..`c98978c` for the full detail, including a fifth, unrelated bug this surfaced in Module 06 (an unmasked-target array-alignment issue that only showed up once the panel stopped being artificially pre-filtered by the old buggy joins).

## Methodological Review (September 2026)

A methodological review of the pipeline — aimed at academic-level scrutiny (arXiv / university presentation) — identified and fixed several issues common to small-N panel studies. The full technical detail lives in each module's docstring and in the commit history from `79a37a6` to `7563671`; the summary here is meant to save a reviewer from re-deriving it:

1. **Standard-error sensitivity is now reported instead of the most favorable single number.** The institutions → FDI coefficient (EQ2) looks "marginally significant" (p ≈ 0.055) only under conventional clustered SEs — the least conservative of the four estimators this pipeline computes, and known to over-reject with G=8 clusters. Under Driscoll-Kraay (p ≈ 0.12) and CR2 Bell-McCaffrey (p ≈ 0.20) it is not significant at 10%; the wild cluster bootstrap gives p ≈ 0.06. See `tables/se_comparison.tex` and Section 11 of the PDF report.
2. **The full transmission chain is now formally tested, not just narrated.** Module 04 bootstraps the product of the three path coefficients (β₁·γ₁·δ₁) by resampling countries. Point estimate: −0.017, 95% CI [−0.079, 0.065], p = 0.62 — the chain as a system is not distinguishable from zero, even though each link's sign matches the theory.
3. **Lag structure is now tested for EQ2 and EQ3, not only EQ1.** The theoretical model argues institutions and growth adjust more slowly than violence, but the primary EQ2/EQ3 specifications used contemporaneous regressors. Re-estimating with one-year lags (Module 05) flips EQ3's sign (+0.137 contemporaneous vs. −0.121 lagged).
4. **A post-treatment ("bad control") check was added.** `gdp_per_capita_log` is plausibly downstream of institutions/FDI, not an exogenous confounder; excluding it moves EQ2's coefficient from 0.900 (p=0.055) to 0.739 (p=0.128).
5. **Country-specific linear trends and a panel unit-root test (Fisher-ADF / Maddala-Wu) were added.** Two-way FE does not remove country-specific trends, and `homicide_rate_log`, `inst_avg` and `gdp_per_capita_log` fail to reject a unit root over this 25-year panel — consistent with EQ1 (violence → institutions) being the least stable link across every check in this pipeline.
6. **Leave-One-Country-Out cross-validation no longer leaks the held-out country into the institution index.** `inst_avg` is rebuilt inside each fold from the training countries' scaling constants only.
7. **A silent bug in the FE-vs-ML convergence table was fixed** — it always printed "?" because of an incorrect JSON key lookup for EQ3's coefficients.

## Panel Extension (September 2026)

Every G=8-related caveat elsewhere in this README speculated that the small number of countries was the binding statistical constraint on this project's results. That hypothesis is now directly testable: Mexico, Ecuador and Peru were added to the panel (11 countries, 274 country-year rows, up from 200), specifically including Mexico for its regional weight in security dynamics (cartel-related violence). The same extraction scripts were reused unchanged in logic — only the country list grew — confirming the data pipeline built during the provenance-fix pass generalizes cleanly.

The effect on results was dramatic, in the direction the small-G caveats predicted:

| | EQ1 (violence→institutions) | EQ2 (institutions→FDI) | EQ3 (FDI→growth) |
|---|---|---|---|
| p (clustered), 8 → 11 countries | 0.352 → **0.093*** | 0.007 → **0.0005*** | 0.306 → **0.010*** |
| p (Driscoll-Kraay) | 0.417 → **0.010*** | 0.085 → **0.003*** | — |
| p (wild cluster bootstrap) | 0.290 → **0.085*** | 0.012 → **0.002*** | 0.450 → 0.108 |

The mediation bootstrap of the full chain's indirect effect moved from a point estimate of -0.0075 (95% CI [-0.069, 0.051], p=0.65) to -0.031 (95% CI [-0.137, 0.015], p=0.144) — the CI still includes zero, but the chain is markedly closer to a formal system-level significance than it was at N=8. Module 07's cointegration test also changed qualitatively: the homicide-rate/GDP-per-capita pair (T3) now shows evidence of cointegration (Fisher-ADF on residuals, p=0.011) where it did not before, and the resulting error-correction model finds a significant speed of adjustment (phi=-0.109, p<0.01): GDP per capita corrects about 11% of any deviation from its long-run relationship with violence each year — the first dynamic (not merely static) causal-adjustment result in this project.

## Key Results Summary

The study is designed to test a sequential mechanism from violence to institutions to investment and then growth, and — on the 11-country panel — every link in that chain now carries the theoretically expected sign and clears conventional significance thresholds under most of the estimators this pipeline reports. EQ2 (institutions → FDI) is the strongest link (p<0.01 under three of four estimators); EQ1 and EQ3 are newly significant under clustered and Driscoll-Kraay SEs and marginally so under the more conservative wild cluster bootstrap (EQ1 p=0.085, EQ3 p=0.108). The formal test of the full chain as a system (the country-cluster bootstrap of the product of the three path coefficients) still does not reject the null at 5% (95% CI [-0.137, 0.015]), but is much closer to doing so than at N=8. This is best read as: the individual links are now reasonably well established, while the *chain as a single mediated system* remains suggestive rather than conclusively proven — exactly the kind of distinction a panel of 11 countries (G=11) can start to make that a panel of 8 could not. Lag-structure and country-trend robustness checks were re-run on the 11-country panel too: EQ2 now survives BOTH a one-year-lag specification (p=0.036, previously not significant at N=8) and country-specific trends (p=0.077) — meaningfully more robust than before. EQ3 still flips from significant-positive (contemporaneous) to insignificant-near-zero (lagged), so that link's sensitivity to specification choice persists and a reader should still weigh it before citing a single point estimate. Module 07's cointegration test now supports a genuine long-run equilibrium between violence and income levels (T3), with a significant error-correction speed of adjustment, while violence and institutions (T1) and institutions and income (T2) still show no cointegration evidence.

## Repository Cleanup Notes

The reproducible workflow is centered on the pipeline under [econometric_pipeline/pipeline](econometric_pipeline/pipeline). Legacy exploratory scripts and redundant datasets have been moved to [archive/legacy_scripts](archive/legacy_scripts) and [archive/legacy_data](archive/legacy_data) so the repository root remains focused on the core analysis workflow. The scripts in [data_extraction](data_extraction) are auxiliary and were created to build earlier versions of the panel data. They are not required to run the main analysis pipeline. The essential datasets for reproducibility are the pipeline input and the pipeline-generated outputs in [econometric_pipeline/pipeline](econometric_pipeline/pipeline).

## Code Documentation

| Script | What does this script do? | Inputs → Outputs | Classification | Status |
|---|---|---|---|---|
| [econometric_pipeline/pipeline/run_pipeline.py](econometric_pipeline/pipeline/run_pipeline.py) | Main orchestration script that runs all pipeline modules in sequence and assembles the final report. | Panel data input → figures, tables, JSON outputs, and final PDF report. | Utility | Core |
| [econometric_pipeline/pipeline/01_data_preparation.py](econometric_pipeline/pipeline/01_data_preparation.py) | Loads the panel, checks balance and missingness, constructs the institutional index, and exports the enriched dataset. | Panel-ready CSV → enriched panel, metadata JSON, summary figures. | ETL / Feature Engineering | Core |
| [econometric_pipeline/pipeline/02_panel_estimation.py](econometric_pipeline/pipeline/02_panel_estimation.py) | Estimates the main two-way fixed-effects models for the violence-institutions-FDI-growth chain. | Enriched panel → FE results JSON and panel with predictions. | Econometrics | Core |
| [econometric_pipeline/pipeline/03_diagnostics.py](econometric_pipeline/pipeline/03_diagnostics.py) | Runs diagnostics for serial correlation, cross-sectional dependence, heteroskedasticity, VIF, and influence. | Panel with predictions → diagnostic JSON and plots. | Econometrics / Visualization | Core |
| [econometric_pipeline/pipeline/04_bootstrap_inference.py](econometric_pipeline/pipeline/04_bootstrap_inference.py) | Implements wild cluster bootstrap inference and CR2 (Bell-McCaffrey, Satterthwaite df) standard errors for small-cluster panels, plus a country-cluster bootstrap test of the full mediation chain's indirect effect and a standard-error sensitivity table. | Panel with predictions → bootstrap results JSON, mediation JSON, SE-comparison LaTeX table, and inference plots. | Econometrics | Core |
| [econometric_pipeline/pipeline/05_robustness.py](econometric_pipeline/pipeline/05_robustness.py) | Tests sensitivity to leave-one-country-out exclusion, time windows, alternative lags (all three links, not only violence→institutions), alternative institution indices, control sets (including a post-treatment "bad control" check), country-specific linear trends, and a Fisher-ADF panel unit-root test. | Panel with predictions → robustness JSON and plots. | Econometrics / Visualization | Core |
| [econometric_pipeline/pipeline/06_ml_triangulation.py](econometric_pipeline/pipeline/06_ml_triangulation.py) | Trains random forest and gradient boosting models for exploratory ML triangulation and feature importance, using leave-one-country-out cross-validation with a leakage-free (fold-safe) institution index. | Panel with predictions → ML results JSON and plots. | ML / Visualization | Core |
| [econometric_pipeline/pipeline/07_cointegration.py](econometric_pipeline/pipeline/07_cointegration.py) | Tests whether the non-stationary variable pairs identified in Module 05 (homicide rate, institutions, GDP per capita) are cointegrated (two-step Engle-Granger/Kao residual-based test) and, where they are, estimates a panel error-correction model separating short-run dynamics from the long-run speed of adjustment. | Panel with predictions → cointegration JSON and residual plots. | Econometrics | Core |
| [econometric_pipeline/pipeline/utils.py](econometric_pipeline/pipeline/utils.py) | Shared helpers for plotting, directory creation, output handling, formatting, and normalizing FE-result JSON lookups across equations (`get_fe_key_stats`). | None → reusable utility functions. | Utility | Core |
| [econometric_pipeline/pipeline/research_report.py](econometric_pipeline/pipeline/research_report.py) | Builds the structured research report used by the pipeline runner. | Text/JSON sections → report objects and PDF-ready content. | Utility | Core |
| [archive/legacy_scripts/01clean_panel.py](archive/legacy_scripts/01clean_panel.py) | Early script for cleaning and preparing a panel-ready dataset from a broader research file. | Raw research dataset → panel-ready CSV. | ETL | Archived (auxiliary) |
| [archive/legacy_scripts/02panel_interpolation.py](archive/legacy_scripts/02panel_interpolation.py) | Explores interpolation and PCA-based dimensionality reduction on the panel dataset. | Panel-ready CSV → PCA-ready dataset. | Feature Engineering | Archived (auxiliary) |
| [archive/legacy_scripts/abc.py](archive/legacy_scripts/abc.py) | Scratch or exploratory script; not used by the main pipeline. | Variable input → no persistent output. | Utility | Archived (auxiliary) |
| [archive/legacy_scripts/build_master_dataset.py](archive/legacy_scripts/build_master_dataset.py) | Builds an intermediate master panel dataset from the combined research file. | Raw merged research file → master panel dataset. | ETL | Archived (auxiliary) |
| [archive/legacy_scripts/crime.py](archive/legacy_scripts/crime.py) | Exploratory visualization script for crime-growth relationships. | Research data → plots. | Visualization | Archived (auxiliary) |
| [archive/legacy_scripts/descriptive_analysis.py](archive/legacy_scripts/descriptive_analysis.py) | Descriptive analysis of the El Salvador subsample and institutional indicators. | Research data → summary tables and descriptive outputs. | Visualization / ETL | Archived (auxiliary) |
| [archive/legacy_scripts/econometrics.py](archive/legacy_scripts/econometrics.py) | Early econometric prototype using linear regression and fitted-value chains. | Panel-ready data → chain-model results. | Econometrics | Archived (auxiliary) |
| [archive/legacy_scripts/econometrics2.py](archive/legacy_scripts/econometrics2.py) | Alternative exploratory econometric script with a simpler OLS structure. | Panel-ready data → summary outputs. | Econometrics | Archived (auxiliary) |
| [archive/legacy_scripts/econometrics3.py](archive/legacy_scripts/econometrics3.py) | Another early econometric script for simple crime-growth analysis. | Panel-ready data → simple regression outputs. | Econometrics | Archived (auxiliary) |
| [archive/legacy_scripts/eda.py](archive/legacy_scripts/eda.py) | Generic exploratory data analysis script for the research dataset. | Research data → descriptive diagnostics and plots. | Visualization | Archived (auxiliary) |
| [archive/legacy_scripts/fe_model.py](archive/legacy_scripts/fe_model.py) | Prototype fixed-effects model using statsmodels. | Panel-ready data → fixed-effects model outputs. | Econometrics | Archived (auxiliary) |
| [archive/legacy_scripts/randomforest.py](archive/legacy_scripts/randomforest.py) | Early random-forest experiment using the panel-ready dataset. | Panel-ready data → feature importance and SHAP outputs. | ML / Visualization | Archived (auxiliary) |
| [data_extraction/data_extractor.py](data_extraction/data_extractor.py) | Downloads macroeconomic indicators from the World Bank API. | API inputs → long-form panel dataset. | ETL | Unused (auxiliary) |
| [data_extraction/merge_panel_datasets.py](data_extraction/merge_panel_datasets.py) | Merges WGI, homicide, and economic datasets into a common panel. | Multiple source datasets → merged panel. | ETL | Unused (auxiliary) |
| [data_extraction/correction_Guatemala.py](data_extraction/correction_Guatemala.py) | Repairs missing GDP per capita observations for Guatemala. | Existing panel data → corrected panel data. | ETL | Unused (auxiliary) |
| [data_extraction/WGI_get.py](data_extraction/WGI_get.py) | Downloads governance indicators from the World Bank API. | API inputs → governance panel. | ETL | Unused (auxiliary) |
| [data_extraction/val.py](data_extraction/val.py) | Lightweight validation script for the research file. | Research data → descriptive validation output. | Utility | Unused (auxiliary) |

## Notes for Reviewers

This repository is structured to support external evaluation by clearly separating the core reproducible workflow from older exploratory scripts. The main analysis can be understood in under ten minutes by following the pipeline entry point and the generated outputs in the pipeline directory. The documentation prioritizes methodological transparency, reproducibility, and a direct mapping between code and empirical claims.
