# Violence, Institutions, and Growth in Latin America and the Caribbean

## Abstract

This repository implements an empirical research workflow for studying how violent crime affects institutional quality and, through that channel, economic growth. The project combines a structured panel-data econometric pipeline with machine-learning triangulation to examine whether the relationship between homicide exposure and growth is direct or mediated by institutional quality and foreign direct investment. The core analysis focuses on a panel of 11 countries — the original 8 in Central America, Colombia and the Dominican Republic, plus Mexico, Ecuador and Peru (added September 2026; see Panel Extension below) — over the period 2000–2024. The empirical strategy is designed to be transparent, reproducible, and suitable for academic review. The repository therefore emphasizes clear data preparation, robust inference, and documented estimation steps rather than ad hoc exploratory analysis.

## Research Question

How does violence, measured by homicide rates, affect institutional quality and subsequent economic performance, and is the relationship between crime and growth mediated by institutions and foreign direct investment?

## Methodology Overview

The project uses a longitudinal panel-data design with country and year effects. The baseline specification is a pooled OLS benchmark, followed by two-way fixed-effects models that absorb time-invariant country heterogeneity and common shocks. Inference is reported with conventional clustered standard errors and with more conservative robust alternatives, including Driscoll-Kraay standard errors to account for cross-sectional dependence, a CR2 (Bell-McCaffrey) bias-corrected estimator with Satterthwaite degrees of freedom, and wild cluster bootstrap inference for small-cluster panels. Because the three-equation chain (violence → institutions → FDI → growth) is estimated as separate single-equation models to avoid a generated-regressors problem, the hypothesized mechanism is additionally tested as a system: a country-cluster bootstrap of the product of the three path coefficients provides a formal test of the indirect effect, rather than relying on each link's significance in isolation. Robustness checks include leave-one-country-out re-estimation, alternative lag structures for every link in the chain (not only violence → institutions), a post-treatment ("bad control") check, country-specific linear trends, and a Fisher-ADF panel unit-root test. In addition, a principal component analysis is used to construct a summary institutional index from the World Governance Indicators, and machine-learning methods are used as an exploratory triangulation layer rather than as a substitute for causal estimation, with a leakage-free (fold-safe) reconstruction of the institution index under Leave-One-Country-Out cross-validation. Finally, an exploratory quantile-regression finding — that violence caps the upper tail of GDP growth without shifting its median — is re-estimated as a hierarchical Bayesian model (MCMC, Asymmetric Laplace likelihood) with partial pooling across countries, reported with full posterior uncertainty rather than asymptotic quantile-regression standard errors.

## Pipeline Architecture

The main entry point of the repository is the econometric pipeline under [econometric_pipeline/pipeline](econometric_pipeline/pipeline). This is the core reproducible system. All steps below are executed through the master runner in [econometric_pipeline/pipeline/run_pipeline.py](econometric_pipeline/pipeline/run_pipeline.py).

The workflow proceeds in nine stages:

1. Data loading: the pipeline reads a clean panel dataset containing country-year observations.
2. Cleaning and ETL: the data are validated, missingness is assessed, and institutional variables are standardized and combined into an index.
3. Feature engineering: the panel is enriched with time trends, lagged violence variables, and derived institutional measures.
4. PCA construction: a robustness institutional index is created using principal component analysis on the governance indicators.
5. Econometric estimation: two-way fixed-effects models estimate the violence-to-institutions, institutions-to-FDI, and FDI-to-growth relationships, each also re-estimated under a one-year-lag alternative.
6. Machine-learning triangulation: random forest and gradient boosting models are trained for exploratory comparison with the econometric estimates, using leave-one-country-out cross-validation with a fold-safe institution index.
7. Validation: diagnostics, bootstrap inference (including a cluster bootstrap of the full mediation chain's indirect effect), and robustness checks assess sensitivity to influential observations, lag choices, control-set choices, country-specific trends, non-stationarity, and model specification.
8. Panel cointegration & error-correction: for the pairs of variables found non-stationary in stage 7 (homicide rate, institutions, GDP per capita), a two-step Engle-Granger/Kao residual-based test checks whether they share a genuine long-run equilibrium before estimating an error-correction model; if not, this is reported as a finding in its own right rather than silently assumed away.
9. Growth-Ceiling-at-Risk (Bayesian MCMC): a hierarchical Bayesian quantile regression (Asymmetric Laplace likelihood, partial pooling across countries) re-estimates an exploratory finding that violence caps the upper tail of GDP growth without affecting its median, replacing LSDV country dummies and asymptotic quantile-regression inference with partial-pooling regularization and full posterior uncertainty.

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
│       ├── 08_growth_ceiling_risk.py
│       ├── 09_synthetic_control.py
│       ├── 10_arch_lm_test.py
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
- World Governance Indicators (WGI): all six Kaufmann et al. (2010) dimensions — rule of law, control of corruption, political stability, voice and accountability, government effectiveness, and regulatory quality
- World Bank remittances data (personal remittances received, % of GDP): tested as a robustness control, not part of the primary specification (see Institution Index Expansion below)
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

The study is designed to test a sequential mechanism from violence to institutions to investment and then growth. Results depend materially on two specification choices this README documents in full: the size of the country panel (8 → 11, see Panel Extension) and the construction of the institution index (3 → 6 WGI dimensions, see Institution Index Expansion) — both changes are reported with their before/after numbers rather than presenting only the final specification. **Under the current specification** (11 countries, 6-dimension institution index): EQ2 (institutions → FDI) is the pipeline's most defensible link — significant in its primary specification (p=0.030 clustered, p=0.006 Driscoll-Kraay, p=0.038 bootstrap) though it no longer survives the one-year-lag or country-trend robustness checks it passed under the narrower 3-dimension index. EQ1 (violence → institutions) is **not significant under any estimator** at 6 dimensions, despite being marginally significant at 3 — testing each WGI dimension individually shows `rule_of_law` alone still moves as expected (p=0.037) while `voice_accountability` does not, diluting the composite. EQ3 (FDI → growth) is significant in its contemporaneous specification (p=0.010) but flips to insignificant-near-zero when lagged, a sensitivity that predates and is unrelated to the institution-index change. The formal test of the full chain as a system (the country-cluster bootstrap of the product of the three path coefficients) does not reject the null at 5% (p=0.548) and is, if anything, further from doing so than under the 3-dimension index. **The honest reading**: EQ2 is reasonably well established; EQ1 should now be treated as a working hypothesis rather than an established result, since its apparent significance under the 3-dimension index did not survive a more complete, standard construction of the same measure; and the *chain as a single mediated system* remains unproven. Module 07's cointegration test supports a genuine long-run equilibrium between violence and income levels (T3, unaffected by the institution-index change, since neither series in that pair is `inst_avg`), with a significant error-correction speed of adjustment, while violence and institutions (T1) and institutions and income (T2) still show no cointegration evidence.

## Exploratory Finding: A "Growth-Ceiling" Pattern (September 2026)

Before committing to build a full Growth-at-Risk (GaR) module, a quick exploratory check was run: a pooled quantile regression (Koenker) of `gdp_growth` on `homicide_rate_log_lag1`, with country fixed effects (LSDV) and a linear year trend, at quantiles 0.05–0.95 (N=231, 11 countries). Classic GaR theory (Adrian, Boyarchenko & Giannone, 2019, "Vulnerable Growth") predicts violence should hit the *lower* tail of the growth distribution hardest; this panel shows the opposite:

| Quantile | coef(homicide, lag 1) | p-value |
|---|---|---|
| 0.05 | +2.68 | 0.030** |
| 0.10 | +0.65 | 0.561 |
| 0.25 | +0.18 | 0.690 |
| 0.50 (median) | -0.10 | 0.813 |
| 0.75 | -0.37 | 0.443 |
| 0.90 | **-2.45** | **0.0001***|
| 0.95 | **-2.06** | **0.0069***|

Violence has essentially no effect near the median, but a large, statistically significant *negative* effect on the *upper* tail (q=0.90/0.95): a "growth-ceiling" pattern where high violence does not make bad years worse, but caps how strong a good year can be. The same check using `inst_avg` (institutions) instead of violence as the conditioning variable found nothing at any quantile (all p>0.27) — the ceiling effect appears specific to violence, not institutional quality generally.

Because a single influential country could easily produce this kind of tail result with only 11 clusters, the upper-tail coefficients were stress-tested with a Leave-One-Country-Out check before treating the pattern as real: refitting q=0.90 and q=0.95 with each of the 11 countries excluded one at a time. The coefficient stayed negative and significant in **all 11** leave-one-out fits at both quantiles (q=0.90 range: -1.62 to -3.21, all p<0.03; q=0.95 range: -1.69 to -3.07, weakest case p=0.087 excluding El Salvador) — no sign flips, and excluding Mexico if anything strengthens the effect. The pattern is not an artifact of any single country.

This was, at the time, a scratch check rather than a committed pipeline module — but robust enough to motivate a proper **Growth-Ceiling-at-Risk** extension. It is now implemented as Module 08; see the section immediately below.

## Growth-Ceiling-at-Risk Module (September 2026)

The exploratory pattern above was re-estimated as [Module 08](econometric_pipeline/pipeline/08_growth_ceiling_risk.py): a hierarchical Bayesian quantile regression (Asymmetric Laplace likelihood, MCMC via PyMC/NUTS) instead of LSDV country dummies and asymptotic quantile-regression standard errors. Partial pooling shrinks each country's intercept toward the grand mean by an amount the data determines — regularization LSDV cannot provide with only G=11 clusters — and every quantity is reported as a full posterior (mean, 94% highest-density interval, and P(coefficient < 0 | data)) rather than a p-value from an asymptotic approximation already flagged as unreliable at this sample size elsewhere in this pipeline.

| Quantile | Frequentist coef. (LSDV) | Bayesian posterior mean | 94% HDI | P(β<0 \| data) |
|---|---|---|---|---|
| 0.05 | +2.21 (p=0.040**) | +1.31 | [-0.03, 2.53] | 0.027 |
| 0.10 | +0.25 (p=0.819) | +0.49 | [-0.53, 1.44] | 0.171 |
| 0.25 | +0.76 (p=0.045**) | +0.19 | [-0.46, 0.84] | 0.294 |
| 0.50 (median) | -0.08 (p=0.827) | -0.23 | [-0.77, 0.37] | 0.790 |
| 0.75 | -0.20 (p=0.650) | -0.45 | [-1.21, 0.26] | 0.880 |
| 0.90 | -2.23 (p=0.0005***) | **-1.76** | **[-2.93, -0.59]** | **0.997** |
| 0.95 | -2.06 (p=0.0043***) | **-2.08** | **[-3.02, -1.11]** | **1.000** |

The Bayesian posterior confirms the frequentist pattern at every quantile — the partial-pooling shrinkage moves point estimates somewhat (e.g. q=0.90: -2.23 → -1.76) without changing the qualitative conclusion — and MCMC diagnostics were clean at every quantile (R-hat ≤ 1.004, effective sample size > 1,100, zero divergent transitions across 4 chains × 1,000 post-warmup draws each). A secondary check repeating the same model with `inst_avg` instead of violence at q=0.90/0.95 found no ceiling effect (P(β<0|data) = 0.367 and 0.597 — essentially a coin flip, HDIs wide and centered near zero) — the effect is specific to violence, not a generic feature of any regressor in this panel. (Numbers reflect the 6-dimension institution index; see Institution Index Expansion below.)

**Growth-Ceiling-at-Risk scenario.** Holding country and year at their average levels, the posterior directly answers the applied question: how much lower is the achievable growth ceiling when lagged violence moves from its empirical 10th to 90th percentile?

| Quantile | Ceiling, low violence (p10) | Ceiling, high violence (p90) | Drop | 94% HDI (drop) | P(drop>0 \| data) |
|---|---|---|---|---|---|
| 0.90 | 8.79 pts | 5.52 pts | 3.27 pts | [1.09, 5.45] | 0.997 |
| 0.95 | 10.29 pts | 6.43 pts | 3.86 pts | [2.06, 5.61] | >0.999 (positive in every posterior draw) |

Caveats: this remains exploratory — motivated by a scratch finding, not a pre-registered hypothesis. G=11 is small even for a hierarchical model; partial pooling regularizes but cannot manufacture information the data does not contain, and priors are weakly informative rather than flat. The Asymmetric Laplace likelihood targets one quantile at a time and does not itself guarantee monotonic quantiles in tau — indeed, the central finding is precisely that the effect is *non-monotonic*: null at the median, negative only in the upper tail.

## Institution Index Expansion (September 2026)

The institution index (`inst_avg` / `inst_pca`) previously used only 3 of the 6 Kaufmann et al. (2010) Worldwide Governance Indicators dimensions (rule of law, control of corruption, political stability) — an arbitrary subset rather than the standard construction. It now uses all six, adding voice & accountability, government effectiveness, and regulatory quality. Remittances (% of GDP), a first-order economic channel in this region (El Salvador alone runs ~20-25% of GDP), were added to the data and tested as a robustness control.

**The 6-dimension index is internally coherent**: Cronbach's α = 0.922 (up from a already-adequate but lower value at 3 dimensions), KMO = 0.863, PC1 explains 72.9% of variance across the six dimensions, and every PCA loading is positive (0.31–0.46) — a single "general governance quality" factor clearly underlies all six series, and `inst_avg`/`inst_pca` remain nearly identical (r = 0.9992).

**But the expansion changes EQ1 and EQ2's results materially, and the honest finding is that they are less robust than the 3-dimension index suggested:**

| | EQ1 (violence→institutions) | EQ2 (institutions→FDI) | EQ3 (FDI→growth) |
|---|---|---|---|
| p (clustered), 3→6 dims | 0.093 → 0.462 | 0.030 → 0.030 | 0.010 → 0.010 |
| p (Driscoll-Kraay) | 0.010 → 0.310 | 0.006 → 0.006 | — |
| p (wild cluster bootstrap) | 0.085 → 0.562 | 0.002 → 0.038 | 0.108 → 0.108 |
| Survives 1-year lag? | — | Yes (p=0.036) → **No (p=0.318)** | — |
| Survives country trends? | — | Yes (p=0.077) → **No (p=0.260)** | — |

EQ1 (violence → institutions) is no longer significant under any estimator. Testing each WGI dimension individually explains why: `rule_of_law` alone remains significant (coef=-1.96, p=0.037) and most dimensions point the expected (negative) direction, but `voice_accountability` moves the *wrong* way (coef=+1.72, p=0.339) — diluting the composite average. EQ2 (institutions → FDI) is still significant in its primary (contemporaneous) specification, but **no longer survives the one-year-lag or country-trend robustness checks that it passed at 3 dimensions** — its apparent robustness was partly an artifact of the narrower index. EQ3 is essentially unchanged (its own significance comes from `fdi_percent_gdp`, not `inst_avg`). The mediation bootstrap of the full chain weakens further (indirect effect -0.011, 95% CI [-0.113, 0.020], p=0.548, vs -0.031/p=0.144 at 3 dimensions).

This is reported as a genuine, important caveat rather than smoothed over: **EQ1 was always the weakest, least stable link in this pipeline (see every prior robustness section), and a more complete, standard construction of the institution index shows it does not survive at all** — the violence→institutions link should now be read as a working hypothesis motivating the rest of the chain, not an established result. EQ2 remains the pipeline's most defensible link, but with a narrower evidence base than previously documented.

**Remittances robustness check** (added as a control, not the primary specification, since it is itself a plausible mediator/collider between violence-driven emigration and growth): EQ1 is unaffected (p=0.179, still not significant). EQ2 strengthens substantially when remittances are included (coef 0.611→0.967, p=0.030→0.0013) — plausibly because remittances absorb variance that otherwise confounds the institutions-FDI relationship, though this single variant should not be over-read either. EQ3 is unchanged (p=0.025, was p=0.018).

## The "Bukele Paradox" (September 2026)

Testing each WGI dimension individually (above) surfaced one genuine anomaly worth a dedicated look: `voice_accountability` is the only one of the six dimensions where lagged violence has the *wrong* sign — less violence coinciding with *lower* voice & accountability, rather than higher. El Salvador's own data explains why:

| Year | Homicide rate (per 100k) | Voice & accountability (0–100) |
|---|---|---|
| 2020 | 21.5 | 58.4 |
| 2021 | 17.3 | 53.5 |
| 2022 | 7.9 | 48.8 |
| 2023 | 2.2 | 48.1 |
| 2024 | 1.9 | 45.0 |

From 2021 to 2024 — the period of El Salvador's state-of-exception security crackdown — homicides collapsed *and* voice & accountability fell sharply. The security gain and the civil-liberties cost moved together, not in opposite directions, which is exactly the reverse of what the other five WGI dimensions (and economic theory) predict.

**Is this El Salvador specifically, or a broader regional pattern?** A Leave-One-Country-Out check on `voice_accountability ~ homicide_rate_log_lag1` (same Two-Way FE spec as the rest of EQ1) answers this directly — now a permanent part of [Module 05](econometric_pipeline/pipeline/05_robustness.py) (Robustness 4b), not just a one-off script:

| Sample | coef | p |
|---|---|---|
| Full sample (11 countries) | +1.72 | 0.339 |
| Excl. El Salvador | **-1.48** | 0.449 |
| Excl. any other single country | +1.14 to +2.74 (always positive) | — |

El Salvador is the **only** country whose exclusion flips the sign. Every other country's exclusion leaves the coefficient positive; excluding El Salvador reverses it to the theoretically expected direction (though neither estimate is significant — this is a small-sample descriptive pattern, not a precisely estimated effect either way). The anomalous coefficient in the full sample is not a regional relationship between violence and voice & accountability — it is specifically the El Salvador case.

![The Bukele Paradox](econometric_pipeline/pipeline/figures/14_bukele_paradox.png)

*Left: the LOCO coefficients above, plotted. Right: El Salvador's homicide rate and voice & accountability score, 2000–2024, with the 2021–2024 state-of-exception period shaded.*

This does not invalidate EQ1's finding across the other five WGI dimensions — it sharpens it. The reason `inst_avg`'s composite coefficient washed out under the 6-dimension index (see above) is not that violence has no relationship with institutional quality; it is that one dimension, in one country, during one specific security transformation, moved in the opposite direction from the rest — because that transformation's defining feature *was* trading civil liberties for security. This is a substantive finding about El Salvador's case specifically, directly relevant to this project's motivating question, not a statistical artifact to explain away.

## Synthetic Control: The Bukele Paradox as a Causal Case Study (October 2026)

The LOCO check above is a robustness check, not a causal design — it shows El Salvador alone drives the anomalous sign, but not what `voice_accountability` would have looked like absent the 2021–2024 state-of-exception crackdown. [Module 09](econometric_pipeline/pipeline/09_synthetic_control.py) answers that narrower question with the Synthetic Control Method (Abadie, Diamond & Hainmueller, 2010): a "synthetic El Salvador" is built as a weighted combination of the other 10 countries, chosen to closely track El Salvador's own 2000–2020 pre-treatment path (matched on the full outcome path rather than covariates, to avoid overfitting with only 10 donors), then compared to the real post-2020 path.

**Donor weights:** Peru (0.325), Dominican Republic (0.320), Colombia (0.247), Nicaragua (0.108) — the rest at zero. Pre-treatment fit is tight (RMSPE = 0.79 points, 2000–2020).

| | Actual | Synthetic | Gap |
|---|---|---|---|
| 2024 `voice_accountability` | 45.0 | 56.9 | **−11.9 points** |

**Placebo-in-space inference** (Abadie et al., 2010): the identical procedure is re-run treating each of the other 10 countries as if they were the treated unit, and El Salvador's post/pre-treatment RMSPE ratio is ranked against this placebo distribution.

| Rank | Country | Post/Pre RMSPE ratio |
|---|---|---|
| **1** | **El Salvador (actual case)** | **11.63** |
| 2 | Dominican Republic | 5.64 |
| 3 | Nicaragua | 3.48 |
| 4 | Honduras | 3.46 |
| 5–11 | Colombia, Ecuador, Guatemala, Costa Rica, Panama, Mexico, Peru | 0.33–2.33 |

El Salvador ranks #1 of 11 — the exact randomization p-value is **0.091** (the minimum attainable with 11 units). Restricting the comparison to the 5 countries whose own pre-treatment fit is at least as good as El Salvador's, it still ranks first (p = 0.200).

![Synthetic Control: El Salvador](econometric_pipeline/pipeline/figures/15_synthetic_control_bukele.png)

*Left: El Salvador's actual `voice_accountability` vs. its synthetic counterfactual, 2000–2024. Right: El Salvador's gap (actual − synthetic, red) against the 10 placebo gaps (gray), state-of-exception period shaded.*

This upgrades the Bukele paradox from "El Salvador is the only country whose exclusion flips the sign" (a robustness statement) to "El Salvador's `voice_accountability` decline is, by a wide margin, the most extreme in the region relative to its own counterfactual trajectory" (a quasi-causal case-study statement). With only 10 donor countries this should be read as illustrative rather than a precisely estimated causal effect, but it is a materially stronger claim than the LOCO check alone.

**A natural follow-up — does this panel show volatility clustering worth modeling with GARCH/MS-GARCH?** No: [Module 10](econometric_pipeline/pipeline/10_arch_lm_test.py) runs Engle's ARCH-LM test (per country, Fisher-combined, since the panel cannot be pooled into one time series without creating spurious jumps at country boundaries). The two headline economic links (EQ2, EQ3 residuals) show no evidence of conditional heteroskedasticity (p=0.292, p=0.215). EQ1 residuals and the homicide-rate series do show some signal — concentrated in a handful of countries (Ecuador, Mexico, Peru for homicide-rate changes; most countries for EQ1, consistent with a common time-varying noise level in the WGI index rather than country-specific regime-switching) — but with T~20–24 annual observations per country, there is nowhere near enough data to fit even a standard GARCH(1,1) reliably, let alone a Markov-Switching GARCH. This is reported as a diagnostic, not acted on: the panel's existing clustered/Driscoll-Kraay standard errors already account for the kind of error heteroskedasticity detected here.

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
| [econometric_pipeline/pipeline/08_growth_ceiling_risk.py](econometric_pipeline/pipeline/08_growth_ceiling_risk.py) | Re-estimates the exploratory "growth-ceiling" quantile pattern with a hierarchical Bayesian quantile regression (ALD likelihood, MCMC via PyMC, partial pooling across countries), benchmarked against the frequentist LSDV quantile regression, plus a Growth-Ceiling-at-Risk scenario (posterior growth ceiling under low vs. high violence). | Panel with predictions → Bayesian quantile-regression JSON, LaTeX table, and figures. | Bayesian Econometrics | Core |
| [econometric_pipeline/pipeline/09_synthetic_control.py](econometric_pipeline/pipeline/09_synthetic_control.py) | Builds a "synthetic El Salvador" from a weighted combination of the other 10 countries to estimate the counterfactual `voice_accountability` path absent the 2021-2024 state-of-exception crackdown, with placebo-in-space inference (Abadie, Diamond & Hainmueller, 2010). | Enriched panel → synthetic-control JSON and figure. | Econometrics (Causal) | Core |
| [econometric_pipeline/pipeline/10_arch_lm_test.py](econometric_pipeline/pipeline/10_arch_lm_test.py) | Engle's ARCH-LM test (per country, Fisher-combined) for conditional heteroskedasticity in the FE residuals and raw series, to check whether a GARCH/MS-GARCH volatility model would have anything to estimate. | Enriched panel + FE residuals → ARCH-LM JSON. | Diagnostics | Standalone (not in main PDF report) |
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
