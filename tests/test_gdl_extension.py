"""Tests for the geometric-deep-learning extension: data, validation design (no leakage), graphs, metrics, verdict."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GDL = Path(__file__).resolve().parents[1] / "gdl_extension"
sys.path.insert(0, str(GDL))
import gdl_common as G  # noqa: E402

pytestmark = pytest.mark.skipif(not (GDL / "data" / "global_panel.csv").exists(), reason="GDL data not built")


@pytest.fixture(scope="module")
def primary():
    return G.load_dataset("primary")


def test_coverage_report_matches_the_coverage_table():
    cov = pd.read_csv(GDL / "data" / "coverage_by_country.csv").set_index("iso3")
    rep = json.load(open(GDL / "data" / "coverage.json"))
    assert rep["n_economies_any_data"] == len(cov)
    core = [f"n_{v}" for v in ("gdp_growth", "log_gdp_pc", "fdi_percent_gdp", "inflation", "exports_percent_gdp", "log_population")]
    assert rep["n_with_core_20plus"] == int((cov[core].min(axis=1) >= 20).sum())


def test_primary_sample_follows_the_stated_coverage_rule(primary):
    d, feats = primary
    cov = pd.read_csv(GDL / "data" / "coverage_by_country.csv").set_index("iso3")
    iso = sorted(d["iso3"].unique())
    assert len(iso) == 92
    assert (cov.loc[iso, "n_log_homicide"] >= 15).all()
    assert (cov.loc[iso, [f"n_{w}" for w in G.WGI]].min(axis=1) >= 20).all()
    assert "log_homicide" in feats and d["target"].notna().all()


def test_target_is_next_year_growth(primary):
    d, _ = primary
    panel = pd.read_csv(GDL / "data" / "global_panel.csv").set_index(["iso3", "year"])["gdp_growth"]
    for _, r in d.sample(50, random_state=0).iterrows():
        assert r["target"] == pytest.approx(panel.loc[(r["iso3"], int(r["year"]) + 1)])
        assert r["target_year"] == r["year"] + 1


@pytest.mark.parametrize("fold", G.FOLDS)
def test_folds_have_no_leakage(primary, fold):
    d, _ = primary
    fit, val, train_all, test = G.fold_indices(d, fold)
    t0 = fold[0]
    assert set(fit).isdisjoint(val) and set(fit) | set(val) == set(train_all)
    assert d.loc[train_all, "target_year"].max() <= t0 - G.GAP          # no training target at or after the test block
    assert d.loc[test, "year"].between(*fold).all()
    assert d.loc[train_all, "year"].max() < t0 - G.GAP                  # a feature year is also left out
    assert d.loc[val, "target_year"].min() > d.loc[fit, "target_year"].max()
    assert len(val) > 0 and len(test) > 0


def test_winsor_bounds_use_only_the_training_window(primary):
    d, _ = primary
    fit, val, train_all, test = G.fold_indices(d, G.FOLDS[-1])
    lo, hi = G.winsor_bounds(d.loc[train_all, "target"])
    assert lo == pytest.approx(np.quantile(d.loc[train_all, "target"], 0.01)) and lo < hi


def test_imputer_uses_training_rows_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [1.0, 1.0, 1.0, 1.0]})
    other = pd.DataFrame({"a": [np.nan, 100.0], "b": [5.0, np.nan]})
    tr, (te,) = G.impute(train, [other])
    assert te.loc[0, "a"] == 2.0 and te.loc[1, "b"] == 1.0              # medians come from training rows, not from `other`
    assert "a_missing" in tr.columns and "b_missing" not in tr.columns
    assert list(te["a_missing"]) == [1.0, 0.0]


def test_graphs_are_row_normalised_and_sparse(primary):
    d, _ = primary
    iso = sorted(d["iso3"].unique())
    graphs = G.build_graphs(iso)
    assert set(graphs) == {"border", "inv_distance", "knn4", "trade"}
    for name, W in graphs.items():
        assert W.shape == (len(iso), len(iso)) and np.allclose(np.diag(W), 0)
        s = W.sum(axis=1)
        assert np.all((np.isclose(s, 1)) | (np.isclose(s, 0))), name     # islands (no land border) have empty rows
        if name in ("inv_distance", "trade"):
            assert (W > 0).sum(axis=1).max() <= G.TOPK
    assert (graphs["knn4"] > 0).sum(axis=1).max() == 4
    b = graphs["border"] > 0
    assert np.array_equal(b, b.T)                                        # land borders are symmetric


def test_node_permutation_preserves_structure():
    rng = np.random.default_rng(0)
    W = np.random.default_rng(1).random((12, 12)) * (np.random.default_rng(2).random((12, 12)) > 0.7)
    P = G.permute_graph(W, rng)
    assert sorted(P.sum(axis=1)) == pytest.approx(sorted(W.sum(axis=1)))
    assert not np.allclose(P, W)


def test_demeaning_and_bootstrap_helpers():
    y = np.array([1.0, 3.0, 10.0, 14.0])
    yr = np.array([2020, 2020, 2021, 2021])
    assert np.allclose(G.demean_by_year(y, yr), [-1, 1, -2, 2])
    loss = np.random.default_rng(0).random(60)
    countries = np.repeat(np.arange(12), 5)
    diff, lo, hi = G.block_bootstrap_diff(loss, loss, countries, B=200)
    assert diff == 0 and lo == 0 and hi == 0
    diff, lo, hi = G.block_bootstrap_diff(loss * 2, loss, countries, B=200, root=False)
    assert lo > 0 and diff == pytest.approx(loss.mean())                  # doubling the loss is a clear, positive difference


def test_neighbour_features_use_only_same_year_information(primary):
    d, feats = primary
    iso = sorted(d["iso3"].unique())
    W = G.build_graphs(iso)["knn4"]
    nb = G.neighbour_features(d, feats[:3], W, iso)
    assert nb.shape == (len(d), 3) and nb.notna().all().all()
    # changing another year's data must not change this year's neighbour features
    d2 = d.copy()
    d2.loc[d2["year"] == 2005, feats[0]] += 1000
    nb2 = G.neighbour_features(d2, feats[:3], W, iso)
    assert np.allclose(nb[d["year"] != 2005].to_numpy(float), nb2[d["year"] != 2005].to_numpy(float))


@pytest.mark.skipif(not (GDL / "outputs" / "verdict_primary.json").exists(), reason="verdict not produced")
def test_verdict_is_consistent_with_its_components():
    v = json.load(open(GDL / "outputs" / "verdict_primary.json"))
    n = 0
    for sec in v["targets"].values():
        assert sec["best_non_graph_score"] <= sec["twin_mlp_score"] + 1e-12 or sec["best_non_graph"].startswith("mlp")
        for r in sec["gnn"].values():
            assert r["success"] == (r["criterion_i_beats_both_and_ci_excludes_zero"] and r["criterion_ii_placebo_keeps_at_most_quarter"])
            if r["criterion_i_beats_both_and_ci_excludes_zero"]:
                assert r["per_seed_mean_score"] < min(sec["best_non_graph_score"], sec["best_slx_score"])
                assert all(x["boot90"][1] < 0 for x in r["vs"].values())
            assert r["n_placebo"] >= 1
            n += r["success"]
    assert v["n_successful_settings"] == n
    assert (n == 0) == v["verdict"].startswith("null result")


@pytest.mark.skipif(not (GDL / "outputs" / "baselines_primary.json").exists(), reason="baselines not produced")
def test_baseline_outputs_cover_all_folds_and_models():
    b = json.load(open(GDL / "outputs" / "baselines_primary.json"))
    assert b["folds"] == [list(f) for f in G.FOLDS]
    for m in ("mean", "last_growth", "ridge|none", "rf|none", "gbm|none", "gbmq|none", "linq|none", "rf|trade", "gbmq|inv_distance"):
        assert m in b["metrics"], m
    for name, m in b["metrics"].items():
        assert m["rmse_dm_all"] > 0 and m["rmse_dm_no2020"] > 0
    assert set(b["winsor_bounds_by_fold"]) == {str(f) for f in G.FOLDS}


@pytest.mark.skipif(not (GDL / "outputs" / "verdict_primary.json").exists(), reason="verdict not produced")
@pytest.mark.parametrize("readme", ["README.md", "README_ES.md"])
def test_readme_gdl_numbers_match_the_verdict(readme):
    text = (GDL.parent / readme).read_text(encoding="utf-8")
    v = json.load(open(GDL / "outputs" / "verdict_primary.json"))
    m, c = v["targets"]["mean"], v["targets"]["ceiling"]
    for x in (m["trivial"]["training mean"], m["best_non_graph_score"], m["twin_mlp_score"], c["best_non_graph_score"], c["twin_mlp_score"]):
        assert f"{x:.3f}" in text
