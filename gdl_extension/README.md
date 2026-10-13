# Geometric deep learning extension (design written before any model was fitted)

Question: **does the graph structure between countries improve out-of-sample prediction of next-year GDP growth, and of its upper tail (the "growth ceiling" of Module 08), beyond what the same country-level features give without a graph?**

This is a predictive exercise. It does not estimate causal effects and does not replace the panel models of the main pipeline. It is motivated by the four papers in the project's reading list: Wang et al. (2022, review of GNNs in finance: graph taxonomy), Sellami et al. (2024, GCN/GAT vs random forest on bilateral trade), Araujo et al. (2026, temporal GNN on 183 countries) and Harit et al. (2024, hypergraphs; not used here).

Why a larger panel: 18 nodes cannot support a GNN. The global panel below has about 90 to 140 nodes and 2,000 to 3,500 country-years, which is still small, so the design is conservative (small models, strong baselines, pre-set rules).

## Data (steps 1 and 2, no modelling)

| Item | Source |
|---|---|
| Features | World Bank API: WGI 2025 revision governance scores (six dimensions), GDP growth, GDP per capita, FDI, inflation, exports, unemployment, population, homicide rate; 2000-2024 |
| Node metadata | World Bank country list (region, income group, capital coordinates) |
| Static edges | land borders (mledoze/countries), CEPII GeoDist (distance, contiguity, language, colonial links) |
| Trade edges | CEPII Gravity V202211, `tradeflow_baci`, averaged over **2000-2008** so the graph contains no information from any evaluated year |

Coverage found in step 1 (`data/coverage.json`): 217 economies with any data; 142 with the six core series in at least 20 of 25 years; 141 of those also with all six WGI dimensions in at least 20 years; **92 also with the homicide rate in at least 15 years**.

## Fixed in advance

**Samples.** Primary: the 92 economies with core series, WGI and homicide coverage (violence is the subject of the project). Sensitivity: the 141 economies without requiring homicides (homicide features dropped).

**Target.** Next-year GDP growth, y(i, t+1), from features observed at t. Reported raw and cross-sectionally demeaned (the model's forecast minus its cross-country mean for that year, against actual growth minus its cross-country mean), because global shocks (2009, 2020) are common to all models and swamp the cross-country structure a graph could add.

**Features at t.** lagged growth, log GDP per capita, FDI, inflation, exports, log population, log homicide rate (primary sample), the six WGI scores. Missing feature values are filled with the training-period median of the country (cross-country median if none) and a missing indicator is added; targets are never imputed.

**Validation.** Rolling-origin blocks of test years 2012-2014, 2015-2017, 2018-2020, 2021-2023 (feature years; targets one year later). Each block is trained only on target years before the block, with a one-year gap. Hyper-parameters are chosen on the last three years of each training window from a small grid fixed in advance; the test blocks are never used for tuning. Results are reported with and without the 2020 target (COVID).

**Models.**
1. No graph: training mean, last-year growth, ridge, random forest, gradient boosting, MLP.
2. Graph as features, the econometric analogue (SLX): the same learners with the weighted neighbour average of lagged features and growth added.
3. GNNs: GCN and GAT; temporal GNN (GConvGRU) if time allows. Graphs: trade, inverse distance, four nearest neighbours, borders.
4. Placebo: each GNN re-run on 20 degree-preserving random rewirings of its graph.

Neural models use 5 seeds; mean and standard deviation are reported. Quantile loss at 0.90 is used for the ceiling analysis; mean squared error for the mean forecast.

**Success criterion.** A graph model counts as a success only if, on the primary sample, its pooled out-of-sample RMSE of the demeaned target (i) beats both the best non-graph learned model and the best SLX model, with a country-block bootstrap 90% interval for the difference that excludes zero, and (ii) loses at least three quarters of that gain when the graph is replaced by its rewired placebo. Anything else is reported as a null result. The same rule is applied to the 0.90 pinball loss.

## Deviations log

Changes to the design above, each made before any full-sample result was seen and applied to every model alike.

1. **Clipped target (2026-10-13).** A one-fold smoke test of the baselines showed that single observations dominate the RMSE (Macao 2023: +75%; Macao 2020: -54%; Ukraine 2022: -29%), so that the metric measured one or two points. Training, validation and evaluated test targets of each fold are now clipped to the 1st-99th percentile range of that fold's training window (no test information is used). The unclipped metrics are kept as secondary. This changes the primary metric stated above.
2. **Linear quantile regression grid.** The first grid (penalty 0.01 and 0.1) shrank every coefficient to zero, so the model returned a constant; the grid is now 0, 0.0005 and 0.005.
3. **Training target (2026-10-13).** Models are trained on the clipped target minus its cross-country mean of the same year, the same quantity they are scored on, so that they do not spend capacity on global shocks (2009, 2020) that no country-level feature explains. Non-demeaned metrics were dropped because predictions are on the demeaned scale.
4. **Imputation (2026-10-13).** Missing feature values are filled with the training-window cross-country median of each column (plus a missing indicator), not the country's own median as first written; this is what the code does for all models.
5. **Sparse graphs, GNN training economy (2026-10-13).** Dense graphs made every GNN fit too slow, so the trade and inverse-distance graphs keep each economy's 10 strongest neighbours (10 largest trade partners; 10 nearest with weights proportional to 1/distance); the baselines use the same graphs. The number of epochs of each GNN setting is selected once on the real graph with seed 0 and reused for the other seeds and for the placebo draws; the GNN placebo has 5 draws per graph instead of 20; training uses at most 200 epochs with patience 20.
6. **Placebo definition.** The placebo graphs are node permutations of the real graph (structure, degrees and weights preserved, alignment with the countries destroyed), not edge rewirings; 5 draws per graph for the baselines, 20 for the GNNs.

## Limits accepted from the start

- The WGI scores carry measurement error as large as their within-country variation (Module 14), which caps what any model can predict from them.
- Homicide data are sparse outside the Americas, Europe and Asia; the primary sample is not representative of the world.
- GNN explanations (attention weights, GNNExplainer) are unstable and will not be presented as evidence of channels.
- This is not causal evidence.

## Environment

The GNN code needs PyTorch and PyTorch Geometric, kept out of the main pipeline's requirements (`requirements-gdl.txt`, a separate virtual environment).

## Results

<!-- RESULTS:BEGIN -->
**Verdict: null result: no GNN setting meets the pre-registered criterion.** On the primary sample (92 economies, 1104 out-of-sample country-years in four rolling-origin blocks, 2012–2023), 0 of 16 GNN settings (GCN and GAT × four graphs × mean and ceiling) meet the criterion fixed beforehand.

- **The variables carry little signal.** RMSE of the demeaned next-year growth: training mean 2.923; best learned model without graph (gbm) 2.783, only 4.8% better; 20 learned models beat the mean, none by much.
- **Graphs do not help the mean forecast.** GNNs 2.847–2.930 against 2.783 (best non-graph) and 2.845 (the MLP twin); 4 of 8 GNN settings are significantly worse than the best non-graph model (90% country-block bootstrap). Graph-as-features (SLX) models: the graph helps in 0 of 12 learner–graph pairs, hurts in 5 and is indistinguishable in 7.
- **Growth ceiling (pinball loss at q=0.90).** Best non-graph (gbmq) 1.069, best SLX 1.059, GNNs 1.065–1.216; the bootstrap interval of the GNN-minus-best-non-graph difference includes zero in 7 of 8 settings. The MLP twin is worse (1.181), so the GNNs look better than it in 7 of 8 settings, but the node-permuted placebo graphs reach 1.062–1.151: message passing acts as a regulariser, not as use of the true topology.
- **Sensitivity sample without homicide features (141 economies, 1692 test country-years):** null result: no GNN setting meets the pre-registered criterion; best non-graph 3.073, GNNs 3.198–3.304, training mean 3.300.
- **Closest call (upper tail).** Primary sample: gat trade scores 1.065 against 1.069 for the best non-graph model (difference -0.004, 90% interval [-0.126, +0.128]); its placebo graphs score 1.062 and keep 102% of its gain over the MLP twin (the criterion allows at most 25%). Sensitivity sample: gcn trade scores 1.144 against 1.168 for the best non-graph model (difference -0.029, 90% interval [-0.103, +0.049]); its placebo graphs score 1.174 and keep 49% of its gain over the MLP twin (the criterion allows at most 25%). Neither sample meets the criterion: the intervals include zero and the placebo graphs recover a large share of the gain, so any information that trade partners carry about the upper tail is at most a weak hint.

These are predictive results on a small panel with noisy governance scores (Module 14); they say that country graphs add no detectable out-of-sample information about next-year growth or its upper tail beyond the countries' own features, not that spillovers do not exist. The design, its deviations log and the limits are in [this file](README.md).
<!-- RESULTS:END -->

## Steps

| Script | Purpose | Status |
|---|---|---|
| `01_build_global_panel.py` | global panel, country metadata, static edges, coverage report | done |
| `02_fetch_trade_edges.py` | bilateral trade edges, 2000-2008 | done |
| `gdl_common.py` | shared data, folds, graphs and metrics | done |
| `03_baselines.py` | no-graph, SLX and placebo baselines | done |
| `04_gnn.py` | MLP, GCN and GAT (mean and ceiling), placebo graphs | done |
| `05_verdict.py` | applies the pre-registered success criterion | done |
| `06_figure.py` | verdict figure | done |
| `07_write_results.py` | writes these results into the READMEs | done |