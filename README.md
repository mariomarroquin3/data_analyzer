# Violence, Institutions, and Growth in Central America and the Caribbean

## Abstract

This repository implements an empirical research workflow for studying how violent crime affects institutional quality and, through that channel, economic growth. The project combines a structured panel-data econometric pipeline with machine-learning triangulation to examine whether the relationship between homicide exposure and growth is direct or mediated by institutional quality and foreign direct investment. The core analysis focuses on a panel of countries in Central America, Colombia, and the Dominican Republic over the period 2000–2024. The empirical strategy is designed to be transparent, reproducible, and suitable for academic review. The repository therefore emphasizes clear data preparation, robust inference, and documented estimation steps rather than ad hoc exploratory analysis.

## Research Question

How does violence, measured by homicide rates, affect institutional quality and subsequent economic performance, and is the relationship between crime and growth mediated by institutions and foreign direct investment?

## Methodology Overview

The project uses a longitudinal panel-data design with country and year effects. The baseline specification is a pooled OLS benchmark, followed by two-way fixed-effects models that absorb time-invariant country heterogeneity and common shocks. Inference is reported with conventional clustered standard errors and with more conservative robust alternatives, including Driscoll-Kraay standard errors to account for cross-sectional dependence, a CR2 (Bell-McCaffrey) bias-corrected estimator with Satterthwaite degrees of freedom, and wild cluster bootstrap inference for small-cluster panels. Because the three-equation chain (violence → institutions → FDI → growth) is estimated as separate single-equation models to avoid a generated-regressors problem, the hypothesized mechanism is additionally tested as a system: a country-cluster bootstrap of the product of the three path coefficients provides a formal test of the indirect effect, rather than relying on each link's significance in isolation. Robustness checks include leave-one-country-out re-estimation, alternative lag structures for every link in the chain (not only violence → institutions), a post-treatment ("bad control") check, country-specific linear trends, and a Fisher-ADF panel unit-root test. In addition, a principal component analysis is used to construct a summary institutional index from the World Governance Indicators, and machine-learning methods are used as an exploratory triangulation layer rather than as a substitute for causal estimation, with a leakage-free (fold-safe) reconstruction of the institution index under Leave-One-Country-Out cross-validation.

## Pipeline Architecture

The main entry point of the repository is the econometric pipeline under [econometric_pipeline/pipeline](econometric_pipeline/pipeline). This is the core reproducible system. All steps below are executed through the master runner in [econometric_pipeline/pipeline/run_pipeline.py](econometric_pipeline/pipeline/run_pipeline.py).

The workflow proceeds in seven stages:

1. Data loading: the pipeline reads a clean panel dataset containing country-year observations.
2. Cleaning and ETL: the data are validated, missingness is assessed, and institutional variables are standardized and combined into an index.
3. Feature engineering: the panel is enriched with time trends, lagged violence variables, and derived institutional measures.
4. PCA construction: a robustness institutional index is created using principal component analysis on the governance indicators.
5. Econometric estimation: two-way fixed-effects models estimate the violence-to-institutions, institutions-to-FDI, and FDI-to-growth relationships, each also re-estimated under a one-year-lag alternative.
6. Machine-learning triangulation: random forest and gradient boosting models are trained for exploratory comparison with the econometric estimates, using leave-one-country-out cross-validation with a fold-safe institution index.
7. Validation: diagnostics, bootstrap inference (including a cluster bootstrap of the full mediation chain's indirect effect), and robustness checks assess sensitivity to influential observations, lag choices, control-set choices, country-specific trends, non-stationarity, and model specification.

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

## Methodological Review (September 2026)

A methodological review of the pipeline — aimed at academic-level scrutiny (arXiv / university presentation) — identified and fixed several issues common to small-N panel studies. The full technical detail lives in each module's docstring and in the commit history from `79a37a6` to `7563671`; the summary here is meant to save a reviewer from re-deriving it:

1. **Standard-error sensitivity is now reported instead of the most favorable single number.** The institutions → FDI coefficient (EQ2) looks "marginally significant" (p ≈ 0.055) only under conventional clustered SEs — the least conservative of the four estimators this pipeline computes, and known to over-reject with G=8 clusters. Under Driscoll-Kraay (p ≈ 0.12) and CR2 Bell-McCaffrey (p ≈ 0.20) it is not significant at 10%; the wild cluster bootstrap gives p ≈ 0.06. See `tables/se_comparison.tex` and Section 11 of the PDF report.
2. **The full transmission chain is now formally tested, not just narrated.** Module 04 bootstraps the product of the three path coefficients (β₁·γ₁·δ₁) by resampling countries. Point estimate: −0.017, 95% CI [−0.079, 0.065], p = 0.62 — the chain as a system is not distinguishable from zero, even though each link's sign matches the theory.
3. **Lag structure is now tested for EQ2 and EQ3, not only EQ1.** The theoretical model argues institutions and growth adjust more slowly than violence, but the primary EQ2/EQ3 specifications used contemporaneous regressors. Re-estimating with one-year lags (Module 05) flips EQ3's sign (+0.137 contemporaneous vs. −0.121 lagged).
4. **A post-treatment ("bad control") check was added.** `gdp_per_capita_log` is plausibly downstream of institutions/FDI, not an exogenous confounder; excluding it moves EQ2's coefficient from 0.900 (p=0.055) to 0.739 (p=0.128).
5. **Country-specific linear trends and a panel unit-root test (Fisher-ADF / Maddala-Wu) were added.** Two-way FE does not remove country-specific trends, and `homicide_rate_log`, `inst_avg` and `gdp_per_capita_log` fail to reject a unit root over this 25-year panel — consistent with EQ1 (violence → institutions) being the least stable link across every check in this pipeline.
6. **Leave-One-Country-Out cross-validation no longer leaks the held-out country into the institution index.** `inst_avg` is rebuilt inside each fold from the training countries' scaling constants only.
7. **A silent bug in the FE-vs-ML convergence table was fixed** — it always printed "?" because of an incorrect JSON key lookup for EQ3's coefficients.

## Key Results Summary

The study is designed to test a sequential mechanism from violence to institutions to investment and then growth. Each individual link carries the theoretically expected sign, but none is robustly significant across all four standard-error estimators the pipeline reports, and a formal country-cluster bootstrap test of the full chain (the product of the three path coefficients) does not reject the null of no indirect effect. The results are best read as evidence *consistent with* an indirect, institutionally-mediated channel rather than as a statistically established causal chain — a limitation inherent to a panel of only 8 countries (G=8), and one the pipeline's own inference machinery (Modules 03–05) is specifically designed to surface rather than obscure. Lag-structure and country-trend robustness checks further show that the institutions → FDI and FDI → growth links are sensitive to specification choices that a reader should weigh before citing a single point estimate.

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
