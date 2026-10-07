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
