# Violence, Institutions, and Economic Growth in Latin America and the Caribbean: Evidence from an Extended Panel, with a Case Study of El Salvador's Security Transformation

**Draft manuscript — v0.1 — September 2026**
*Author: Mario Marroquín*

> **Status note (remove before submission):** This draft updates and extends the Spanish-language research paper submitted to Complejo Educativo Thomas Jefferson (Sonsonate, El Salvador) in June 2026, under the *Economía del Desarrollo* research line ("El impacto de la seguridad en el crecimiento económico de El Salvador"). That original paper's introduction, hypothesis, objectives, and literature review are reused and translated below (Sections 1–2); its empirical numbers (an 8-country panel, a 3-dimension institution index) have since been superseded by this project's continued work and are replaced throughout with current results. All empirical numbers below are pulled directly from the pipeline's JSON outputs (not retyped from memory) and are current as of the commits on `main` through the "Bukele paradox" addition. Citations carried over from the original paper (Section 2, marked accordingly) were sourced and reviewed as part of that submission; citations added for the newer methods (panel cointegration, Bayesian quantile regression) were not independently re-verified for this draft and should be checked before submission.

---

## Abstract

This paper studies whether violent crime affects economic growth in Latin America and the Caribbean through its effect on institutional quality and, in turn, foreign direct investment (FDI). Using a panel of 11 countries (Central America, Colombia, the Dominican Republic, Mexico, Ecuador, and Peru; 2000–2024, N=274 country-year observations), we estimate the chain violence → institutions → FDI → growth as three single-equation two-way fixed-effects models, avoiding the generated-regressors problem of estimating it as a single system. Inference accounts explicitly for the panel's small number of clusters (G=11) using four estimators — conventional clustered, Driscoll-Kraay, CR2 Bell-McCaffrey, and wild cluster bootstrap — and the indirect effect of the full chain is tested formally via a country-cluster bootstrap of the product of path coefficients rather than inferred from the significance of each link in isolation.

We document two central findings, both of which depend on specification choices we report transparently rather than omit. First, extending the institution index from an initially-used subset of 3 Worldwide Governance Indicators (WGI) dimensions to the complete, standard 6-dimension Kaufmann et al. (2010) construction — while producing a more internally coherent index (Cronbach's α=0.92, KMO=0.86, PC1 explains 72.9% of variance) — causes the violence → institutions link to lose significance under every inference method (clustered p=0.46, Driscoll-Kraay p=0.31, bootstrap p=0.56), where it had been marginally significant under the narrower 3-dimension index. The institutions → FDI link remains significant in its primary specification (p=0.030 clustered, p=0.006 Driscoll-Kraay) but no longer survives one-year-lag or country-trend robustness checks that it passed before. Second, decomposing the institution index into its six individual dimensions reveals that one — voice and accountability — is the sole dimension where violence has the theoretically wrong sign, and a Leave-One-Country-Out check shows El Salvador is the only country whose exclusion reverses that sign. We document this as a substantive case-study finding, not a statistical artifact: El Salvador's 2021–2024 state-of-exception security crackdown drove homicides from 17.3 to 1.9 per 100,000 while voice and accountability *also* fell (53.5 to 45.0) — security gains and civil-liberties costs moving together rather than in the trade-off direction the composite index otherwise assumes.

As a secondary contribution, we test for panel cointegration among the non-stationary series in the chain and find a genuine long-run equilibrium between violence and income levels (Fisher-ADF residual test, p=0.011) with a significant error-correction speed of adjustment (φ=−0.109, p=0.006) — a dynamic result absent from the static specification. Finally, motivated by an exploratory finding that violence compresses the *upper* tail of the GDP growth distribution without affecting its median or lower tail, we estimate a hierarchical Bayesian quantile regression (Asymmetric Laplace likelihood, MCMC via NUTS) and find that moving from the 10th to 90th percentile of lagged violence lowers the achievable 90th-percentile growth ceiling by 3.3 points (94% credible interval [1.1, 5.4], posterior probability of a real drop ≈ 0.997) — a "growth-ceiling" pattern that is the mirror image of the canonical lower-tail Growth-at-Risk framing (Adrian, Boyarchenko & Giannone, 2019).

---

## 1. Introduction

The relationship between security, institutional quality, and economic development has occupied a central place in development economics over recent decades. The institutional economics literature holds that economic growth depends substantially on institutions' capacity to guarantee the rule of law, protect property rights, and generate a predictable environment for investment and productive activity (North, 1990; Rodrik, Subramanian & Trebbi, 2004; Acemoglu & Robinson, 2012). In this view, safer societies tend to offer better conditions for investment, strengthen institutional functioning, and favor sustained economic growth (World Bank, 2011).

More recent research, however, suggests that security is not only a direct determinant of economic performance but also an observable manifestation of a state's institutional capacity to enforce the law, control corruption, and maintain political stability (Kaufmann, Kraay & Mastruzzi, 2010; World Bank, 2011). From this perspective, violence is not merely a public-security problem but a symptom of how well the institutions responsible for legal order and governability are functioning — and can accordingly be read as an indicator of institutional weakness. Countries with high crime rates tend to face greater difficulty enforcing the law, lower institutional trust, and a less favorable environment for investment and productive development (World Bank, 2011; UNODC, 2023). Cross-country evidence shows institutional quality to be among the principal determinants of development differences between countries, even above geography or trade openness (Rodrik, Subramanian & Trebbi, 2004; Acemoglu & Robinson, 2012). Most of this literature, however, studies gradual change over long periods; contemporary cases where security conditions change abruptly are comparatively rare.

El Salvador is one such exceptional case. Between 2015 and 2016 the country recorded one of the world's highest homicide rates, exceeding 100 per 100,000 inhabitants; it subsequently experienced a reduction of more than 95% in under a decade, moving to among the lowest homicide rates in Latin America (UNODC, 2023; World Bank, 2024). The magnitude and speed of this transformation make the country a case of particular interest for studying how institutions and the economy respond to an abrupt change in security conditions.

### 1.1 Research question

*How are security, institutional quality, and economic development related in Latin America and the Caribbean, and what does the case of El Salvador reveal about that transmission mechanism?*

### 1.2 Hypothesis

The central hypothesis holds that the relationship between security and economic growth is predominantly indirect. Specifically, higher levels of violence are hypothesized to deteriorate institutional quality — particularly the rule of law, control of corruption, and political stability — reducing countries' capacity to attract foreign investment and, in consequence, limiting economic performance (North, 1990; World Bank, 2011; Acemoglu & Robinson, 2012). These effects are further hypothesized to operate over different time horizons: security conditions can change relatively quickly in response to public policy, while institutions evolve more gradually, and economic effects require still longer periods to fully reflect these transformations. Under this view, growth would not respond immediately to a reduction in violence, but through the progressive strengthening of institutions and a more favorable investment environment. Formally, the hypothesized four-stage transmission mechanism is:

$$\underbrace{H_{it}}_{\text{Violence}} \xrightarrow{\beta_1<0} \underbrace{Q_{it}}_{\text{Institutions}} \xrightarrow{\gamma_1>0} \underbrace{F_{it}}_{\text{FDI}} \xrightarrow{\delta_1>0} \underbrace{G_{it}}_{\text{GDP Growth}}$$

where *H* is the homicide rate, *Q* institutional quality, *F* foreign direct investment, and *G* economic growth, for country *i* and year *t*. The signs above each arrow are the directions the hypothesis predicts.

### 1.3 Objectives

**General objective.** To analyze the relationship between security, institutional quality, and economic growth in Latin America and the Caribbean using a panel-data econometric approach, complemented by machine-learning techniques, in order to identify the mechanisms through which security influences economic performance — taking El Salvador as the reference case study.

**Specific objectives.**
1. Construct a panel dataset through the extraction, cleaning, integration, and transformation of data from international organizations, ensuring the consistency, comparability, and quality of the economic, institutional, and security variables used.
2. Design and implement an analytical pipeline integrating mathematical, econometric, and machine-learning techniques to model the relationships between security, institutional quality, and economic growth, incorporating validation procedures, robust inference, and sensitivity analysis.
3. Analyze and interpret the results obtained through the pipeline, contrasting them with institutional-economics theory and available empirical evidence, in order to draw conclusions about the transmission mechanisms and the differing adjustment speeds between security, institutions, and economic growth.

**Contribution.** Relative to the original (June 2026) version of this study, this draft's empirical contribution is threefold: (i) the panel is extended from 8 to 11 countries (adding Mexico, Ecuador, and Peru) and the institution index from 3 to 6 Worldwide Governance Indicators dimensions, the standard construction in the literature, and both extensions are shown to materially change which links in the hypothesized chain are statistically supported — reported transparently rather than silently adopted; (ii) a case-study finding — the "Bukele paradox" — isolates exactly which country and which institutional dimension is responsible for an otherwise-puzzling composite-index result, turning what could have been reported as noise into a specific, falsifiable claim about El Salvador's security transformation; and (iii) two methodological extensions beyond the original study's scope — panel cointegration/error-correction modeling and a hierarchical Bayesian Growth-Ceiling-at-Risk analysis — test, respectively, whether a genuine long-run equilibrium underlies the chain's static estimates, and whether violence affects the upper tail of the growth distribution differently from its center.

**Roadmap.** Section 2 reviews the relevant literature. Section 3 describes the data and the institution index. Section 4 lays out the empirical strategy. Section 5 reports the main results and their robustness. Section 6 discusses the institution-index-construction sensitivity finding in detail. Section 7 presents the Bukele-paradox case study. Section 8 presents the Growth-Ceiling-at-Risk extension. Section 9 discusses limitations. Section 10 concludes.

---

## 2. Literature Review

*(This section reuses and organizes the literature review from the original June 2026 submission, translated to English. Citations marked † were added for the methodological extensions introduced after that submission and were not part of the original review; they should be treated as less thoroughly vetted and re-checked independently before submission.)*

**Institutions and economic development.** The foundational claim motivating this paper — that institutions are a first-order determinant of economic performance — follows North (1990), who frames institutions as the rules of the game that structure economic incentives, and Acemoglu & Robinson (2012), who argue that inclusive versus extractive institutions explain much of the cross-country variation in long-run prosperity. Rodrik, Subramanian & Trebbi (2004) provide the cross-country empirical anchor for this claim, showing institutional quality dominates geography and trade integration as a predictor of income differences across countries — the basis for this paper's focus on institutions as the central mediating channel rather than a direct violence-growth link.

**Violence as an institutional symptom.** Kaufmann, Kraay & Mastruzzi (2010), who define the Worldwide Governance Indicators this paper's institution index is built from, and the World Bank's *World Development Report 2011: Conflict, Security and Development* together motivate treating violence not merely as a public-security outcome but as an observable symptom of a state's institutional capacity to enforce law, control corruption, and maintain political stability. The UNODC's *Global Study on Homicide* (2023) is this paper's other principal source for framing the scale and trajectory of homicide in the region, including El Salvador's transformation specifically.

**Bidirectional endogeneity between violence and institutions.** Besley & Persson (2011) document that the relationship between weak institutions and violence plausibly runs in both directions: weak institutions can facilitate higher crime, while high violence also erodes institutional capacity. This simultaneity concern is the central reason this paper uses lagged violence, two-way fixed effects, and (Section 5.4) an explicit test for reverse dynamics via panel cointegration, rather than relying on a single bivariate or contemporaneous specification.

**Inference with a small number of clusters.** Cameron & Miller (2015) is the methodological basis for this paper's approach to standard errors under a small number of country clusters (G=11); Driscoll & Kraay (1998) is the basis for the Driscoll-Kraay HAC estimator used throughout as a check robust to cross-sectional dependence specifically.

**Machine learning as triangulation, not causal inference.** Breiman (2001) is the methodological reference for the random forest models used in Section 5.5 as a non-parametric robustness check on the linear fixed-effects results' variable-importance ranking; the companion gradient boosting models follow the same logic.

**El Salvador's informal economy as a transmission channel.** *Diario El Mundo* (2024) reports that informal employment constitutes roughly 70% of El Salvador's workforce — cited in Section 9 as a plausible reason security gains might first manifest as reduced extortion and operating costs for existing informal and small-formal businesses, before appearing in aggregate FDI or GDP growth figures.

**† Small-sample panel cointegration.** Engle & Granger (1987), extended to panels by Kao (1999) and McCoskey & Kao (1998), and the Fisher-type panel unit-root test of Maddala & Wu (1999), underlie Section 5.4 and Section 7's test for whether the chain's static relationships reflect a genuine long-run equilibrium rather than a potentially spurious levels regression.

**† Growth-at-Risk and Bayesian quantile regression.** Adrian, Boyarchenko & Giannone (2019) introduce the Growth-at-Risk framework this paper's Section 8 inverts (studying the upper rather than lower tail of the growth distribution); Yu & Moyeed (2001) and the non-centered hierarchical parameterization of Betancourt & Girolami (2015) and Gelman & Hill (2007) underlie the Bayesian estimation approach used there.

**What remains to be done.** This review, inherited from the original submission and lightly extended, has not been independently re-searched for this draft. In particular, a fuller pass should still cover: the broader cross-country empirical literature on violence, investment, and FDI specifically (beyond the institutional-quality channel emphasized here); critiques of aggregating multiple WGI dimensions into a single index (directly relevant to Section 6's finding that this choice matters); and the growing literature — largely outside economics — on El Salvador's *régimen de excepción* specifically, which Section 7 engages with empirically but without yet citing that literature directly.

---

## 3. Data

### 3.1 Panel construction

The panel covers 11 countries — El Salvador, Guatemala, Honduras, Nicaragua, Costa Rica, Panama, the Dominican Republic, Colombia, Mexico, Ecuador, and Peru — over 2000–2024 (274 country-year observations; unbalanced, T=25 distinct years, mean ≈24.9 years per country). The original 8-country panel (Central America, Colombia, the Dominican Republic) was extended to add Mexico, Ecuador, and Peru, specifically including Mexico for its regional weight in organized-crime-related violence. All data come from the World Bank's World Development Indicators and Worldwide Governance Indicators APIs, pulled programmatically (not from a manually-curated file) so the extraction is fully reproducible; homicide data is the World Bank's mirror of UNODC crime statistics, with two recent El Salvador observations (2023–2024) patched from public reporting because the live WDI mirror had not yet caught up as of extraction (see `data_extraction/merge_panel_datasets.py` for the documented patch).

### 3.2 Institution index

The institution index (`inst_avg`) is the equal-weighted standardized average of all six Kaufmann et al. (2010) Worldwide Governance Indicators dimensions — rule of law, control of corruption, political stability and absence of violence, voice and accountability, government effectiveness, and regulatory quality — each z-scored on the pooled panel before averaging. A PCA-based robustness index (`inst_pca`, first principal component of the same six z-scored variables) is nearly identical (r=0.999). The index is internally coherent: Cronbach's α=0.922, Kaiser-Meyer-Olkin sampling adequacy=0.863, and the first principal component explains 72.9% of the pooled variance across the six dimensions, with every PCA loading positive (0.31–0.46). Section 6 discusses why this specification choice — six dimensions rather than a narrower subset — matters for the paper's results.

### 3.3 Key variables

| Variable | Description | Transformation |
|---|---|---|
| `homicide_rate_log_lag1` | Homicides per 100,000, one-year lag | log(1+x) |
| `inst_avg` / `inst_pca` | Composite institution index (6 WGI dimensions) | standardized average / PC1 |
| `fdi_percent_gdp` | Foreign direct investment, % of GDP | level |
| `gdp_growth` | Real GDP growth rate, % | level |
| `gdp_per_capita_log` | GDP per capita (current US$) | log |
| `remittances_percent_gdp` | Personal remittances received, % of GDP | level (control, not primary spec — see Section 6.3) |

Controls used throughout: `gdp_per_capita_log`, `inflation`, `exports_percent_gdp`, `population_log`, `unemployment`.

---

## 4. Empirical Strategy

### 4.1 Baseline specification

The hypothesized mechanism — violence → institutions → FDI → growth — is estimated as three **separate** single-equation two-way fixed-effects models rather than a single simultaneous system, specifically to avoid a generated-regressors problem (using a first-stage fitted value as a second-stage regressor, which biases second-stage standard errors if not corrected for):

- **EQ1:** `inst_avg_it = α_i + λ_t + β₁·homicide_rate_log_lag1_it + controls_it + ε_it`
- **EQ2:** `fdi_percent_gdp_it = α_i + λ_t + γ₁·inst_avg_it + controls_it + ε_it`
- **EQ3:** `gdp_growth_it = α_i + λ_t + δ₁·fdi_percent_gdp_it + δ₂·inst_avg_it + controls_it + ε_it`

All estimated via `PanelOLS` with entity and time fixed effects.

### 4.2 Inference under a small number of clusters

With G=11 clusters, conventional cluster-robust standard errors are known to be anti-conservative (over-reject; Cameron & Miller, 2015). Every coefficient of interest is therefore reported under four estimators: (i) conventional clustered SE; (ii) Driscoll-Kraay HAC, robust to cross-sectional dependence; (iii) CR2 Bell-McCaffrey bias-corrected clustered SE with Satterthwaite degrees of freedom; (iv) wild cluster bootstrap with Webb (2023) six-point weights (999 replications). A coefficient described as "robust" in this paper means robust across all four, not significant under the most favorable one.

### 4.3 Testing the chain as a system

Because the three equations are estimated separately, no single test statistic directly evaluates the full chain. We construct one: a country-cluster bootstrap (resampling countries with replacement, 500 replications) of the product of the three path coefficients, `indirect = β₁·γ₁·δ₁`, following standard mediation-analysis logic but adapted to the panel-cluster-bootstrap setting.

### 4.4 Panel cointegration and error-correction

A Fisher-type panel unit-root test (Maddala & Wu, 1999; per-country Augmented Dickey-Fuller, combined via Fisher's method) is run on every series in the chain. For pairs that both fail to reject a unit root (i.e., behave as I(1)), we test for cointegration using the two-step Engle-Granger approach extended to panels (Engle & Granger, 1987; Kao, 1999; McCoskey & Kao, 1998): estimate the long-run relationship via two-way FE, then test the residuals for stationarity with the same Fisher-ADF test. Where residuals are stationary, we estimate an error-correction model separating the short-run effect of a change in x from the long-run speed of adjustment.

### 4.5 ML triangulation

Random forest and gradient boosting models are trained on each of the three tasks as a non-parametric triangulation of the FE results' feature-importance ranking (SHAP values, permutation importance, Gini importance), using Leave-One-Country-Out cross-validation with a fold-safe reconstruction of the institution index (recomputing z-score scaling constants from training countries only, to avoid the held-out country's observations leaking into the index it is being evaluated against). This is explicitly **not** a substitute for the causal FE estimates; its role is to check whether the key variables in each equation are consistently important to a flexible, non-parametric model, independent of the linear functional form the FE specification assumes.

### 4.6 Growth-Ceiling-at-Risk (Bayesian)

Section 8 estimates a hierarchical Bayesian quantile regression via the Asymmetric Laplace Distribution likelihood (Yu & Moyeed, 2001), with a non-centered partial-pooling prior on country intercepts (Betancourt & Girolami, 2015; Gelman & Hill, 2007) in place of the LSDV country dummies a frequentist quantile regression would require. This targets the full posterior distribution of the coefficient at each quantile — including a directly interpretable posterior probability P(β<0 | data) — rather than an asymptotic p-value, which is particularly relevant at G=11.

---

## 5. Results

### 5.1 Main estimates

| | EQ1: violence → institutions | EQ2: institutions → FDI | EQ3: FDI(+inst.) → growth |
|---|---|---|---|
| Key coefficient | −0.084 | 0.611 | 0.207 (FDI); −0.544 (inst.) |
| SE (clustered) | 0.114 | 0.280 | 0.080 (FDI) |
| p (clustered) | 0.462 | 0.030 | 0.010 (FDI) |
| p (Driscoll-Kraay) | 0.310 | 0.006 | — |
| p (CR2 Bell-McCaffrey) | 0.602 | 0.108 | 0.093 (FDI) |
| p (wild cluster bootstrap) | 0.562 | 0.038 | 0.108 |
| N | 231 | 264 | 264 |
| Within R² | 0.203 | 0.019 | 0.077 |

EQ2 (institutions → FDI) is the only link significant under a majority of the four estimators; EQ1 (violence → institutions) is not significant under any of them at the current (6-dimension) institution-index specification — see Section 6 for why this differs from an earlier, narrower specification of the same index. EQ3's FDI coefficient is significant under three of four estimators; its institutions coefficient (−0.544, included as a direct as well as mediated channel) is not (p=0.337 clustered), consistent with institutions' effect on growth running primarily *through* FDI rather than directly.

### 5.2 Robustness

**Leave-One-Country-Out.** EQ1's coefficient ranges from −0.005 to −0.302 across the 11 leave-one-out refits — always negative, never flipping sign, though the point estimate varies by a factor of 60.

**Lag structure (EQ1).** Contemporaneous: β=−0.022 (p=0.861). Lag 1 (primary): β=−0.084 (p=0.462). Lag 2: β=−0.211 (p=0.112). Lag 3: β=−0.306 (p=0.027) — the coefficient grows more negative and more significant at longer lags, consistent with the theoretical prior that institutional erosion from violence operates with a delay, but this pattern should be read cautiously: it was not the pre-specified primary lag, and testing four lag lengths without correction inflates the chance of finding one significant result by construction.

**Time windows (EQ1).** Pre-2010: β=−0.124 (p=0.360, N=76). Post-2010: β=−0.013 (p=0.822, N=155). Excluding COVID years (2020–2021): β=−0.055 (p=0.593, N=210). Post-"peace" (2014+): β=+0.027 (p=0.626, N=113, wrong sign). No sub-window recovers significance; the point estimate is unstable in both magnitude and, in the most recent sub-period, sign.

**Control sets.** EQ1's coefficient is stable in sign and magnitude (−0.08 to −0.16) across minimal, baseline, excl.-GDP-per-capita, and incl.-remittances control sets, and never reaches significance in any of them except "extended controls" (tourist arrivals and trade added: β=−0.213, p=0.0021) — a variant that drops N from 231 to 191 due to tourist-arrival data missingness, so this apparent significance should not be over-weighted without first checking whether it survives on the common (N=191) sample under the baseline control set too. EQ2's coefficient similarly survives most control-set variants (0.13 to 0.97) except "extended controls" (β=−1.07, p=0.277, wrong sign, N=216 vs. 264 baseline) — again confounded with the N drop. Including remittances strengthens EQ2 substantially (β=0.611→0.967, p=0.030→0.0013); we discuss why this should not be read as establishing remittances as a legitimate control, rather than a coincidentally variance-absorbing one, in Section 6.3.

**Country-specific linear trends.** EQ1: β=−0.069 (p=0.157, still n.s.). EQ2: β=0.910 (p=0.260, loses the significance it has at baseline). EQ3 (FDI): β=0.190 (p=0.075, weakens from p=0.010 but remains marginally significant).

**Placebo test.** Randomly permuting `homicide_rate_log_lag1` within year cells 500 times and comparing the true EQ1 coefficient (−0.084) against the resulting null distribution (mean=0.002, SD=0.024) gives p=0.000 — the true coefficient is more extreme than every placebo draw. This is in evident tension with the clustered-SE result (p=0.462) reported above, and we flag rather than resolve that tension here: the placebo test's within-year permutation preserves less of the panel's cross-sectional and serial correlation structure than the cluster-robust SE accounts for, so the two tests are not measuring the same null and should not be expected to agree. We report both rather than select the one that tells a cleaner story.

**Panel unit-root test (Fisher-ADF).** `homicide_rate_log` (p=0.582), `inst_avg` (p=0.242), and `gdp_per_capita_log` (p=0.861) all fail to reject a unit root (consistent with I(1) behavior); `fdi_percent_gdp` (p<0.001) and `gdp_growth` (p<0.001) both reject (consistent with I(0)). This motivates the cointegration analysis in Section 5.4 for the I(1) pairs only.

### 5.3 Testing the chain as a system

The country-cluster bootstrap of the indirect effect `β₁·γ₁·δ₁` gives a point estimate of −0.011 (bootstrap mean −0.015, SD 0.030), 95% CI [−0.113, 0.020], p=0.548. The formal test of the chain as a mediated system does not reject the null that the indirect effect is zero.

### 5.4 Cointegration

| Pair | Long-run β | Fisher-ADF p (residuals) | Cointegrated? |
|---|---|---|---|
| T1: Homicide ↔ Institutions | −0.133 | 0.636 | No |
| T2: Institutions ↔ Income | 0.773 | 0.393 | No |
| T3: Homicide ↔ Income | −0.091 | 0.011 | **Yes** |

For T3, the error-correction model finds φ=−0.109 (p=0.006 clustered, p<0.001 Driscoll-Kraay): income corrects roughly 11% of any deviation from its long-run equilibrium with violence each year. The short-run effect (γ=−0.033, p=0.344) is not significant — only the long-run adjustment is. T1 and T2 show no cointegration evidence, consistent with (and offering an additional formal explanation for) EQ1's instability across every other check in this paper.

### 5.5 ML triangulation

Random forest and gradient boosting LOCO-CV R² is strongly negative for T1 (RF −28.8, GB −26.3) and T2 (RF −1.27, GB −2.05) — both models generalize worse across held-out countries than a naive mean predictor, consistent with EQ1 and EQ2's fragility in the linear FE results. T3 (FDI + institutions → growth) is the only task with LOCO-CV R² above zero (RF 0.059, GB 0.227). Feature-importance rankings (SHAP) place the key variable of interest in the top 4 of each task (T1: rank 4, T2: rank 3, T3: rank 1), offering some support for each variable's relevance even where the LOCO-CV predictive performance itself is poor — a distinction (importance-for-explanation vs. out-of-sample predictive power) we are careful not to conflate.

---

## 6. The Institution Index: Construction Matters

### 6.1 Three dimensions versus six

An earlier version of this analysis constructed `inst_avg` from 3 of the 6 Kaufmann et al. (2010) WGI dimensions (rule of law, control of corruption, political stability) — a subset chosen without a stated methodological justification, essentially a carryover from an earlier, manually-assembled data file. Recognizing this as an arbitrary rather than principled choice, we reconstructed the index using all six dimensions, the standard construction in the literature. The results reported in Section 5 use this 6-dimension index throughout.

The two specifications are not a matter of indifference:

| | EQ1 (p, clustered) | EQ1 (p, Driscoll-Kraay) | EQ1 (p, bootstrap) | EQ2 (p, clustered) | EQ2 (p, bootstrap) |
|---|---|---|---|---|---|
| 3-dimension index | 0.093 | 0.010 | 0.085 | 0.030 | 0.002 |
| 6-dimension index | 0.462 | 0.310 | 0.562 | 0.030 | 0.038 |

Under 3 dimensions, EQ1 was marginally significant under three of four estimators. Under 6 dimensions — the more complete, standard construction — it is not significant under any of them. EQ2 remains significant at 6 dimensions but, as Section 5.2 reports, no longer survives lag-structure or country-trend robustness checks that it passed at 3 dimensions.

### 6.2 Why: a per-dimension decomposition

Re-estimating EQ1 with each of the six WGI dimensions individually as the dependent variable (instead of the composite) shows the source of the attenuation directly:

| Dimension | Coefficient | p (clustered) |
|---|---|---|
| Rule of law | −1.961 | **0.037** |
| Control of corruption | −0.395 | 0.645 |
| Political stability | −2.672 | 0.135 |
| Voice & accountability | **+1.721** | 0.339 |
| Government effectiveness | −0.999 | 0.253 |
| Regulatory quality | −0.277 | 0.803 |

Five of six dimensions point in the theoretically expected (negative) direction; rule of law remains individually significant. One dimension — voice and accountability — points the wrong way, with a coefficient large enough in magnitude to materially dilute the composite average. Section 7 investigates this dimension specifically and traces it to a single country.

### 6.3 A note on the remittances robustness check

Remittances (% of GDP) were added to the dataset as a candidate control, motivated by their economic importance in the region (El Salvador alone runs ≈20–25% of GDP in remittances). We deliberately did **not** add them to the primary control set: remittances are plausibly a mediator or collider between violence-driven emigration and growth (households funded by relatives who emigrated, in part, because of violence), which is a structurally different concern from an omitted confounder. Including them as a robustness variant shows EQ2 strengthening substantially (β 0.611→0.967, p 0.030→0.0013) and EQ1 essentially unaffected (β −0.155, p=0.179, still n.s.). We report this pattern but do not treat it as evidence that remittances "should" be a control — doing so would risk exactly the post-treatment bias this paper is careful to flag for `gdp_per_capita_log` elsewhere (Section 5.2).

---

## 7. Case Study: The "Bukele Paradox"

### 7.1 The anomaly

Section 6.2 isolated `voice_accountability` as the one WGI dimension where lagged violence has the theoretically wrong sign. El Salvador's own trajectory shows why:

| Year | Homicide rate (per 100,000) | Voice & accountability (0–100) |
|---|---|---|
| 2020 | 21.5 | 58.4 |
| 2021 | 17.3 | 53.5 |
| 2022 | 7.9 | 48.8 |
| 2023 | 2.2 | 48.1 |
| 2024 | 1.9 | 45.0 |

From 2021 to 2024 — the period of the state-of-exception security crackdown — homicides collapsed while voice and accountability *also* fell. The security gain and the civil-liberties cost moved together, not in opposing directions, which is the reverse of what the other five WGI dimensions (and, implicitly, a composite index that treats all six as interchangeable proxies for "institutional quality") assume.

### 7.2 Is this El Salvador, or the region?

A Leave-One-Country-Out check on `voice_accountability ~ homicide_rate_log_lag1` (identical Two-Way FE specification to the rest of EQ1) tests whether the anomaly reflects a broader regional relationship or is specific to a single country:

| Sample | Coefficient | p |
|---|---|---|
| Full sample (11 countries) | +1.721 | 0.339 |
| Excl. El Salvador | **−1.482** | 0.449 |
| Excl. any other single country | +1.14 to +2.74 (always positive) | — |

El Salvador is the **only** country whose exclusion flips the sign. Excluding any of the other ten leaves the coefficient positive, ranging from +1.14 to +2.74; excluding El Salvador specifically reverses it to the theoretically expected direction (though, with N=10 countries remaining, neither estimate is itself significant — this is a small-sample descriptive result, not a precisely estimated effect in either specification).

### 7.3 Interpretation

This is not read here as a statistical artifact to be explained away, but as a substantive finding about El Salvador's case specifically, directly relevant to the question motivating this paper. The reason the composite institution index's relationship with violence weakens under the complete 6-dimension construction (Section 6.1) is not that violence has no relationship with institutional quality in this panel — it is that one dimension, in one country, during one specific and historically unusual security transformation, moved in the opposite direction from the other five, because that transformation's defining and most contested feature *was* trading civil liberties for security. A composite index built to treat "institutional quality" as a single latent construct is, by design, not well suited to a case where two of its components are moving in opposite directions for the same underlying reason. This does not invalidate the index or the paper's broader EQ1 finding across the other five dimensions; it sharpens what that finding does and does not capture, and it turns El Salvador from an unexplained outlier into the paper's most specific and most falsifiable claim.

*(Figure: `econometric_pipeline/pipeline/figures/14_bukele_paradox.png` — Leave-One-Country-Out forest plot and El Salvador's homicide-rate / voice-and-accountability time series, 2000–2024, state-of-exception period highlighted.)*

---

## 8. Growth-Ceiling-at-Risk: A Bayesian Exploration

### 8.1 Motivation

An exploratory frequentist quantile regression of `gdp_growth` on lagged violence (country fixed effects via dummy variables, quantiles 0.05–0.95) found violence has essentially no effect near the median of the growth distribution but a large, negative, Leave-One-Country-Out-robust effect on the *upper* tail (q=0.90/0.95) — the mirror image of the canonical Growth-at-Risk framing (Adrian, Boyarchenko & Giannone, 2019), which targets the lower tail. High violence, on this pattern, does not make bad years worse; it caps how good a good year can be.

### 8.2 A hierarchical Bayesian re-estimation

We re-estimated this pattern as a hierarchical Bayesian quantile regression (Asymmetric Laplace likelihood; Yu & Moyeed, 2001) with partial pooling across the 11 countries (non-centered parameterization; Betancourt & Girolami, 2015) in place of LSDV country dummies, targeting the full posterior distribution at each quantile rather than an asymptotic p-value.

| Quantile (τ) | Frequentist coef. | Bayesian posterior mean | 94% HDI | P(β<0 \| data) |
|---|---|---|---|---|
| 0.05 | +2.21 (p=0.040) | +1.31 | [−0.03, 2.53] | 0.027 |
| 0.50 (median) | −0.08 (p=0.827) | −0.23 | [−0.77, 0.37] | 0.790 |
| 0.90 | −2.23 (p=0.0005) | **−1.76** | **[−2.93, −0.59]** | **0.997** |
| 0.95 | −2.06 (p=0.0043) | **−2.08** | **[−3.02, −1.11]** | **1.000** |

Partial pooling regularizes the point estimates somewhat (e.g., at q=0.90, −2.23 → −1.76) without changing the qualitative conclusion. MCMC diagnostics are clean at every quantile (R-hat ≤ 1.004, effective sample size > 1,100, zero divergent transitions across 4 chains × 1,000 post-warmup draws). A secondary check repeating the same model with `inst_avg` instead of violence at q=0.90/0.95 finds no ceiling effect (P(β<0|data)=0.367 and 0.597 — indistinguishable from a coin flip) — the upper-tail effect is specific to violence, not a generic feature of any regressor in this panel.

### 8.3 An applied risk quantity

Holding country and year at their average levels, the posterior directly answers the applied question: how much lower is the achievable growth ceiling when lagged violence moves from its empirical 10th to 90th percentile?

| Quantile | Ceiling, low violence (p10) | Ceiling, high violence (p90) | Drop | 94% HDI (drop) | P(drop>0\|data) |
|---|---|---|---|---|---|
| 0.90 | 8.79 pts | 5.52 pts | 3.27 pts | [1.09, 5.45] | 0.997 |
| 0.95 | 10.29 pts | 6.43 pts | 3.86 pts | [2.06, 5.61] | >0.999 |

### 8.4 Caveats

This remains exploratory, motivated by a post-hoc pattern rather than a pre-registered hypothesis. G=11 is small even for a hierarchical model: partial pooling regularizes but cannot manufacture information the data does not contain, and the priors are weakly informative rather than flat. The Asymmetric Laplace likelihood estimates each quantile independently and does not itself guarantee monotonic quantiles in τ — indeed, the central finding here is precisely that the effect is *non-monotonic*: null at the median, negative only in the upper tail.

---

## 9. Limitations

1. **Identification and bidirectional endogeneity.** Every result in this paper is a conditional association under two-way fixed effects, not a causal estimate in the design-based sense. Lagging violence by one year and absorbing country and year fixed effects addresses some, not all, reverse-causality concerns; as Besley & Persson (2011) document, weak institutions and violence plausibly run in both directions (weak institutions permit crime; crime erodes institutions), and this paper's design narrows rather than eliminates that simultaneity. The panel cointegration test in Section 5.4 is a partial response — a genuine long-run equilibrium is harder to generate from a purely spurious or purely reverse-causal relationship — but it is not a full resolution.
2. **Sample size.** G=11 clusters is small for every inference method used, including the more conservative ones. Several results (Sections 5.1–5.2) depend visibly on which of four standard-error estimators is quoted; we report all four throughout specifically so no single favorable number can be cited in isolation.
3. **Index construction sensitivity.** Section 6 demonstrates that a seemingly minor choice — which subset of a standard governance index to use — changes which links in the causal chain appear statistically significant. This should be read as a caution against over-interpreting any single specification of the institution index, including the 6-dimension one used as primary throughout this paper.
4. **The Bukele-paradox case study is a single country.** Section 7's finding is, by construction, about one country's experience during one specific period. It is offered as a substantive, specific, falsifiable claim — not as a basis for generalizing about security-liberty trade-offs across the region.
5. **The Growth-Ceiling-at-Risk result (Section 8) is exploratory**, discovered via quantile regression rather than derived from an ex ante hypothesis, and should be treated as motivation for future, pre-registered work rather than a confirmed finding.
6. **The informal-economy transmission channel (Section 1, Section 10) is asserted, not tested.** The claim that security gains first reduce extortion and operating costs for El Salvador's large informal sector (≈70% of employment; *Diario El Mundo*, 2024) before appearing in aggregate FDI or growth is a plausible mechanism consistent with this paper's lag-structure results, not a directly estimated effect — this project's data does not include firm-level or informal-sector-specific series.
7. **Literature review remains partial** (Section 2). It reuses and lightly extends the original submission's review; the items flagged there as not yet covered should be addressed before this paper is positioned as a complete contribution to any of the literatures it touches.

---

## 10. Conclusion

This paper's honest summary is not "violence causes institutional decline, which suppresses FDI and growth" — the data, estimated rigorously and reported in full rather than selectively, do not support that clean a story, and did not even in the original June 2026 submission, where EQ1's p-value (0.303) and EQ3's (0.213) were already far from conventional significance under a smaller, 8-country panel. What the data support, now with a larger panel and a more complete institution index, is narrower and, we think, more useful: institutions → FDI is the panel's most defensible link; violence → institutions is fragile and specification-dependent, appearing marginally significant under an arbitrary 3-dimension index construction and disappearing entirely under the standard 6-dimension one; and where the aggregate relationship is puzzling — one governance dimension moving the "wrong" way — decomposing it reveals a specific, interpretable, case-level explanation (El Salvador's voice-and-accountability decline during the state of exception) rather than unexplained noise.

The original submission's central interpretive contribution — that security, institutions, and growth operate on different adjustment speeds, with security able to shift quickly through policy while institutions and macroeconomic aggregates respond with a lag — remains, in our view, the right lens for reading these results, and is reinforced rather than undermined by the extensions in this draft. EQ1's coefficient grows more negative at longer lags (Section 5.2); a genuine long-run equilibrium between violence and income levels exists (Section 5.4), with income correcting roughly a ninth of any deviation each year — a dynamic, adjustment-speed result the original paper's static specification could not have produced. And an exploratory but LOCO-robust pattern — violence compressing the upper tail of achievable growth without touching the median (Section 8) — motivates a specific, falsifiable direction for future, pre-registered work. Understanding these differences in response speed matters directly for policy evaluation: the benefits of a security improvement appear to depend on, and only fully materialize through, the institutional strengthening that accompanies or follows it — and, as El Salvador's case specifically shows, that institutional strengthening is not guaranteed to be uniform across every dimension of what "institutional quality" is usually taken to mean.

---

## References

**From the original June 2026 submission** (sourced and reviewed as part of that submission):

- Acemoglu, D., & Robinson, J. A. (2012). *Why Nations Fail: The Origins of Power, Prosperity, and Poverty*. Crown Business.
- Besley, T., & Persson, T. (2011). *Pillars of Prosperity: The Political Economics of Development Clusters*. Princeton University Press.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.
- Diario El Mundo (2024). El Salvador con cerca del 70% de empleo informal, una de las tasas más altas de América Latina. Retrieved from https://diario.elmundo.sv/economia/el-salvador-con-cerca-del-70-de-empleo-informal-una-de-las-tasas-mas-altas-de-america-latina
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560.
- Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators: Methodology and Analytical Issues. *World Bank Policy Research Working Paper* No. 5430.
- North, D. C. (1990). *Institutions, Institutional Change and Economic Performance*. Cambridge University Press.
- Rodrik, D., Subramanian, A., & Trebbi, F. (2004). Institutions rule: The primacy of institutions over geography and integration in economic development. *Journal of Economic Growth*, 9(2), 131–165.
- United Nations Office on Drugs and Crime (2023). *Global Study on Homicide 2023*. UNODC.
- World Bank (2011). *World Development Report 2011: Conflict, Security and Development*. World Bank.
- World Bank (2024). World Development Indicators. Retrieved from https://databank.worldbank.org/source/world-development-indicators
- World Bank (2024). Worldwide Governance Indicators. Retrieved from https://info.worldbank.org/governance/wgi/

**Added for the methodological extensions in this draft** (not independently re-verified — see status note):

- Adrian, T., Boyarchenko, N., & Giannone, D. (2019). Vulnerable Growth. *American Economic Review*, 109(4), 1263–1289.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
- Bartlett, M. S. (1954). A note on the multiplying factors for various χ² approximations. *Journal of the Royal Statistical Society, Series B*, 16(2), 296–298.
- Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical models. In *Current Trends in Bayesian Methodology with Applications*.
- Brodeur, A., Cook, N., & Heyes, A. (2020). Methods matter: p-hacking and publication bias in causal analysis in economics. *American Economic Review*, 110(11), 3634–3660.
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Kaiser, H. F. (1970). A second generation little jiffy. *Psychometrika*, 35(4), 401–415.
- Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. *Journal of Econometrics*, 90(1), 1–44.
- Maddala, G. S., & Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. *Oxford Bulletin of Economics and Statistics*, 61(S1), 631–652.
- McCoskey, S., & Kao, C. (1998). A residual-based test of the null of cointegration in panel data. *Econometric Reviews*, 17(1), 57–84.
- Oster, E. (2019). Unobservable selection and coefficient stability: theory and evidence. *Journal of Business & Economic Statistics*, 37(2), 187–204.
- Webb, M. D. (2023). Reworking wild bootstrap-based inference for clustered errors. *Canadian Journal of Economics*, 56(3), 839–858.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Yu, K., & Moyeed, R. A. (2001). Bayesian quantile regression. *Statistics & Probability Letters*, 54(4), 437–447.
- Yu, K., & Zhang, J. (2005). A three-parameter asymmetric Laplace distribution and its extension. *Communications in Statistics — Theory and Methods*, 34(9–10), 1867–1879.

---

## Appendix A: Reproducibility

All results in this paper are generated by a version-controlled, fully reproducible pipeline: [github.com/mariomarroquin3/data_analyzer](https://github.com/mariomarroquin3/data_analyzer). Every number quoted above can be traced to a specific module (`econometric_pipeline/pipeline/01_data_preparation.py` through `08_growth_ceiling_risk.py`) and a specific JSON output file. The full technical report, including every diagnostic and robustness check not reproduced in this manuscript, is `econometric_pipeline/pipeline/econometric_report.pdf`, regenerated automatically by `run_pipeline.py` from the same pipeline that produced this paper's numbers.
