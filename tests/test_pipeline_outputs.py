"""Invariants of the pipeline's JSON outputs."""
import math


def test_metadata_matches_panel(load_json, panel):
    meta = load_json("01_metadata.json")
    assert meta["N"] == panel["country_code"].nunique() == len(meta["countries"])
    assert meta["N_obs"] == len(panel)


def test_fe_pvalues_are_valid_probabilities(load_json):
    fe = load_json("02_fe_results.json")
    for eq in ("EQ1", "EQ2"):
        assert 0.0 <= fe[eq]["pval_key_cl"] <= 1.0
        assert 0.0 <= fe[eq]["pval_key_dk"] <= 1.0
    for p in fe["EQ3"]["pvals_cl"].values():
        assert 0.0 <= p <= 1.0


def test_mediation_ci_contains_point_estimate(load_json):
    med = load_json("04_mediation.json")
    assert med["ci_lo"] <= med["indirect_full"] <= med["ci_hi"]
    assert 0.0 <= med["p_boot"] <= 1.0


def test_synthetic_control_weights_and_inference(load_json, meta):
    sc = load_json("09_synthetic_control.json")
    w = sc["el_salvador"]["weights"]
    assert all(v >= 0 for v in w.values())
    # Weights below 1e-4 are dropped on export, so the sum is ~1.
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-3)
    assert "SLV" not in w                       # treated unit is not its own donor
    assert sc["n_countries"] == len(sc["placebo_ratios"]) <= meta["N"]
    assert sc["el_salvador"]["n_donors"] == sc["n_countries"] - 1 - len(sc["el_salvador"]["dropped_donors"])
    # Randomization p-value is exactly rank / n.
    assert math.isclose(sc["p_value"], sc["rank_of_slv"] / sc["n_countries"], rel_tol=1e-9)
    # Reported ratio equals post/pre RMSPE.
    e = sc["el_salvador"]
    assert math.isclose(e["ratio"], e["post_rmspe"] / e["pre_rmspe"], rel_tol=1e-6)
    assert math.isclose(e["gap_2024"], e["actual_2024"] - e["synthetic_2024"], abs_tol=1e-9)


def test_arch_lm_pvalues_are_valid(load_json, meta):
    arch = load_json("10_arch_lm_test.json")
    for res in arch.values():
        assert 0.0 <= res["p_combined"] <= 1.0
        assert res["n_countries"] <= meta["N"]


def test_spec_curve_consistent_with_main_estimates(load_json):
    """The primary (all-6-dimension) specification in the spec curve must
    reproduce Module 02's EQ1/EQ2 estimates exactly."""
    sp = load_json("11_spec_curve.json")
    fe = load_json("02_fe_results.json")
    assert len(sp["specs"]) == 63                       # 2**6 - 1 non-empty subsets
    for eq in ("EQ1", "EQ2"):
        prim = sp["summary"][eq]["primary_6dim"]
        assert math.isclose(prim["coef"], fe[eq]["coef_key_cl"], rel_tol=1e-6)
        assert math.isclose(prim["p_cl"], fe[eq]["pval_key_cl"], rel_tol=1e-6)
        s = sp["summary"][eq]
        assert 0.0 <= s["share_sig_clustered_05"] <= 1.0
        assert s["coef_min"] <= s["coef_median"] <= s["coef_max"]


def test_synthetic_control_robustness_block_present(load_json):
    rb = load_json("09_synthetic_control.json")["robustness"]
    for key in ("leave_one_donor_out", "placebo_in_time", "excluding_drift_donors",
                "onset_2022", "first_stage_homicide"):
        assert key in rb
    lo, hi = rb["leave_one_donor_out_gap_range"]
    assert lo <= hi


def test_spatial_diagnostics_outputs_are_valid(load_json, meta):
    sp = load_json("12_spatial_diagnostics.json")
    sc = load_json("09_synthetic_control.json")
    assert sp["n_countries"] == meta["N"]
    for res in sp["pesaran_cd"].values():
        assert 0.0 <= res["p"] <= 1.0
    for res in sp["morans_i"].values():
        assert 0.0 < res["p_perm"] <= 1.0              # permutation p-values are never exactly 0
    for res in sp["slx"].values():
        assert 0.0 < res["p_wild"] <= 1.0
    s = sp["summary"]
    assert s["n_slx"] == len(sp["slx"]) and s["n_moran"] == len(sp["morans_i"])
    # the contamination check must read the same weights as Module 09
    w = sc["el_salvador"]["weights"]
    chk = sp["synthetic_control_contamination"]
    assert math.isclose(chk["weight_on_land_neighbours"], sum(w.get(c, 0.0) for c in chk["land_neighbours"]), abs_tol=1e-9)


def test_fast_wild_bootstrap_matches_panelols(panel):
    """utils.fe_wild_test must reproduce PanelOLS two-way FE point estimates."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometric_pipeline" / "pipeline"))
    from linearmodels.panel import PanelOLS
    from utils import fe_wild_test
    d = panel.set_index(["country_code", "year"])
    exog = ["homicide_rate_log_lag1", "gdp_per_capita_log", "inflation"]
    w = d[["homicide_rate_log"] + exog].dropna()
    ref = PanelOLS(w["homicide_rate_log"], w[exog], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    out = fe_wild_test(w, "homicide_rate_log", "homicide_rate_log_lag1", exog, B=199)
    assert math.isclose(out["coef"], float(ref.params["homicide_rate_log_lag1"]), rel_tol=1e-8)
    assert 0.0 < out["p_wild"] <= 1.0


def test_causal_link_scan_is_internally_consistent(load_json):
    ls = load_json("13_causal_link_scan.json")
    s = ls["summary"]
    links = ls["links"]
    assert ls["n_links"] == len(links) == 72                      # 9 variables, ordered pairs
    assert s["n_raw_sig_05"] == sum(L["primary"]["p_wild"] < 0.05 for L in links.values())
    assert s["n_bh_survivors"] == sum(L["bh_survives"] for L in links.values())
    assert s["n_holm_survivors"] <= s["n_bh_survivors"] <= s["n_raw_sig_05"]
    for L in links.values():
        assert 0.0 < L["primary"]["p_wild"] <= 1.0
        if L["bh_survives"]:
            assert L["primary"]["p_wild"] <= s["fdr_q"]
    # every reported chain must be built from two surviving links
    surv = {(L["cause"], L["effect"]) for L in links.values() if L["bh_survives"]}
    for c in ls["chains"]:
        x, m, y = [v.strip() for v in c["chain"].split("->")]
        assert (x, m) in surv and (m, y) in surv


def _pipeline_utils():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometric_pipeline" / "pipeline"))
    import utils
    return utils


def test_link_design_placebo_keeps_real_lagged_outcome_control():
    """The placebo scan replaces only the CAUSE; the lagged-outcome control must stay the real series."""
    u = _pipeline_utils()
    dep, key, exog = u.link_design("a", "b", ["a", "b", "c"], lag_y=True, cause_suffix="__plc")
    assert dep == "b" and key == "a__plc"
    assert "b__lag1" in exog and "b__plc" not in exog          # control is not shuffled
    assert exog.count("a__plc") == 1 and "a" not in exog and "b" not in exog and "c" in exog
    _, key2, exog2 = u.link_design("a", "b", ["c"], lag_y=False)
    assert key2 == "a__lag1" and "b__lag1" not in exog2


def test_country_cluster_bootstrap_resamples_whole_countries(panel):
    u = _pipeline_utils()
    seen = []

    def stat(d):
        seen.append(d.index.get_level_values(0).nunique())
        return float(d["gdp_growth"].mean())

    draws = u.country_cluster_bootstrap(panel, stat, B=25, seed=1)
    assert len(draws) == 25 and all(n == panel["country_code"].nunique() for n in seen)   # copies get unique labels
    assert draws.std() > 0


def test_chain_indirect_effect_bootstrap_runs_on_real_links(panel):
    """Smoke test of the machinery behind Module 13's chain step (no chain survives on current data)."""
    u = _pipeline_utils()
    d = panel.copy()
    for c in ("homicide_rate_log", "inst_avg", "fdi_percent_gdp"):
        d[c + "__lag1"] = d.groupby("country_code")[c].shift(1)
    ctrl = ["gdp_per_capita_log", "inflation"]

    def link(x, y, data):
        dep, key, exog = u.link_design(x, y, ctrl)
        w = data[[dep] + exog].dropna()
        return u.fe_wild_test(w, dep, key, exog, B=1)["coef"]

    draws = u.country_cluster_bootstrap(
        d, lambda data: link("homicide_rate_log", "inst_avg", data) * link("inst_avg", "fdi_percent_gdp", data), B=10, seed=3)
    assert len(draws) >= 8


def test_wild_bootstrap_agrees_with_module_04(load_json, panel):
    """The shared helper and Module 04 use different p-value conventions; the gap on EQ1 must stay small."""
    u = _pipeline_utils()
    meta = load_json("01_metadata.json")
    ctrl = [c for c in meta["controls"] if c != "year_c"]
    d = panel.set_index(["country_code", "year"])
    exog = ["homicide_rate_log_lag1"] + ctrl
    w = d[["inst_avg"] + exog].dropna()
    out = u.fe_wild_test(w, "inst_avg", "homicide_rate_log_lag1", exog, B=999)
    ref = load_json("04_bootstrap.json")["eq1"]
    assert abs(out["coef"] - ref["beta_full"]) < 1e-8
    assert abs(out["p_wild"] - ref["p_boot"]) < 0.06


def test_spatial_adjacency_definitions_are_symmetric_and_consistent(load_json):
    sp = load_json("12_spatial_diagnostics.json")
    adj = sp["weights"]["adjacent_pairs"]
    assert adj["contiguity"] == sp["weights"]["land_borders"]
    assert set(adj) == {"contiguity", "knn4", "within_1500km"}
    assert adj["knn4"] >= 18 * 4 // 2                           # a symmetrised 4-NN graph has at least n*k/2 edges
    for key, v in sp["neighbour_pair_gap"].items():
        an = key.split("_", 1)[1]
        assert 0 < v["n_adjacent_pairs"] <= adj[an]
