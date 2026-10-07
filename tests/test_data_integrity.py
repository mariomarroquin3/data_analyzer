"""Invariants of the analysis panel (econometric_pipeline/pipeline/panel_enriched.csv)."""
import numpy as np

from conftest import WGI_COLS


def test_panel_shape_and_countries(panel, meta):
    assert set(panel["country_code"]) == set(meta["countries"])
    assert len(panel) == meta["N_obs"]
    assert meta["N"] == len(meta["countries"]) >= 11
    assert "SLV" in set(panel["country_code"])          # the case-study country
    assert panel["year"].min() == 2000 and panel["year"].max() == 2024


def test_no_duplicate_country_years(panel):
    assert not panel.duplicated(["country_code", "year"]).any()


def test_country_names_are_valid_utf8(panel):
    # Guards against mojibake (e.g. "MÃ©xico") if the CSV is ever re-encoded.
    names = set(panel["country_name"])
    assert not any("Ã" in n or "�" in n for n in names)
    assert {"México", "Panamá", "Perú"} <= names


def test_wgi_dimensions_are_percentile_ranks(panel):
    for col in WGI_COLS:
        s = panel[col].dropna()
        assert s.between(0, 100).all(), col


def test_homicide_rate_non_negative_and_log_consistent(panel):
    h = panel[["homicide_rate", "homicide_rate_log"]].dropna()
    assert (h["homicide_rate"] >= 0).all()
    assert np.allclose(h["homicide_rate_log"], np.log1p(h["homicide_rate"]), atol=1e-6)


def test_institution_index_requires_all_six_dimensions(panel):
    """inst_avg and inst_pca must be defined exactly when all six WGI
    dimensions are observed (regression test for the partial-NaN bug)."""
    complete = panel[WGI_COLS].notna().all(axis=1)
    assert (panel["inst_avg"].notna() == complete).all()
    assert (panel["inst_pca"].notna() == complete).all()


def test_el_salvador_recent_homicides_match_official_figures(panel):
    """2023: 2.4 (154 homicides, Fiscalia General); 2024: 1.9 (114 homicides)."""
    slv = panel[panel["country_code"] == "SLV"].set_index("year")["homicide_rate"]
    assert slv.loc[2023] == 2.4
    assert slv.loc[2024] == 1.9
    assert slv.loc[2024] < slv.loc[2022] < slv.loc[2021]
