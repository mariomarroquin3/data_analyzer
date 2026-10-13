"""Guards against documentation drifting away from the pipeline outputs.

Headline numbers in the READMEs are typed by hand; these tests fail when a
pipeline re-run changes a number that the documentation still quotes.
The paper/ checks run only where paper/ exists (it is kept out of the
public repository).
"""
import re

import pytest

from conftest import PIPE, ROOT


def _read(path):
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("readme", ["README.md", "README_ES.md"])
def test_readme_headline_numbers_match_json(readme, load_json):
    text = _read(ROOT / readme)
    fe = load_json("02_fe_results.json")
    sc = load_json("09_synthetic_control.json")
    assert f"{abs(sc['el_salvador']['gap_2024']):.1f}" in text
    assert f"{sc['p_value']:.3f}" in text
    assert f"{sc['el_salvador']['ratio']:.2f}" in text
    assert f"{fe['EQ1']['pval_key_cl']:.3f}" in text or f"{fe['EQ1']['pval_key_cl']:.2f}" in text


def test_report_text_has_no_hardcoded_panel_size():
    """The PDF report text must derive the country count from the data
    (via NC), not hard-code it."""
    src = _read(PIPE / "run_pipeline.py")
    assert not re.search(r"G\s*=\s*\d+", src.replace("G={NC}", ""))
    assert not re.search(r"\d+ países", src)


def test_module_docstrings_match_current_cluster_count(meta):
    """Every 'G=<n>' / 'N=<n> countries' mentioned in module docstrings and the
    pipeline README must equal the current number of countries."""
    n = meta["N"]
    files = sorted(PIPE.glob("0[2-8]_*.py")) + [PIPE / "README.md"]
    for f in files:
        for m in re.finditer(r"\b[GN]\s*=\s*(\d+)\b", _read(f)):
            assert int(m.group(1)) == n, f"{f.name}: '{m.group(0)}' but the panel has {n} countries"


@pytest.mark.skipif(not (ROOT / "paper" / "manuscript.md").exists(), reason="paper/ not present")
def test_manuscript_synthetic_control_numbers_match_json(load_json):
    text = _read(ROOT / "paper" / "manuscript.md")
    sc = load_json("09_synthetic_control.json")
    assert f"{abs(sc['el_salvador']['gap_2024']):.1f}" in text
    assert f"{sc['p_value']:.3f}" in text
    assert f"{sc['el_salvador']['ratio']:.2f}" in text


@pytest.mark.parametrize("readme", ["README.md", "README_ES.md"])
def test_readme_spatial_and_link_scan_numbers_match_json(readme, load_json):
    text = _read(ROOT / readme)
    ls = load_json("13_causal_link_scan.json")
    sp = load_json("12_spatial_diagnostics.json")
    ms, ss = ls["summary"], sp["summary"]
    assert f"{ms['n_raw_sig_05']} de {ls['n_links']}" in text or f"{ms['n_raw_sig_05']} of {ls['n_links']}" in text
    assert f"{ss['n_slx_significant_05']} de {ss['n_slx']}" in text or f"{ss['n_slx_significant_05']} of {ss['n_slx']}" in text
    best = min(ls["links"].values(), key=lambda L: L["primary"]["p_wild"])
    assert f"{best['primary']['p_wild']:.3f}" in text


@pytest.mark.parametrize("readme", ["README.md", "README_ES.md"])
def test_readme_growth_ceiling_numbers_match_json(readme, load_json):
    """Module 08 is an MCMC run: any re-run moves these numbers, so the READMEs must be refreshed with it."""
    text = _read(ROOT / readme).replace("\u2212", "-")
    gc = load_json("08_growth_ceiling_risk.json")
    for q in ("0.90", "0.95"):
        b = gc["quantiles"][q]["bayes"]
        assert f"{b['beta_mean']:.2f}" in text
        assert f"{b['beta_hdi_lo']:.2f}" in text and f"{b['beta_hdi_hi']:.2f}" in text
        assert f"{b['p_beta_negative']:.3f}" in text
        s = gc["growth_ceiling_scenario"][q]
        assert f"{s['ceiling_drop_mean']:.2f} pts" in text
        assert f"{s['ceiling_low_violence_mean']:.2f} pts" in text
    lo = gc["quantiles"]["0.05"]["bayes"]
    assert f"{lo['p_beta_negative']:.3f}" in text


@pytest.mark.parametrize("readme", ["README.md", "README_ES.md"])
def test_readme_wgi_measurement_and_ceiling_robustness_numbers_match_json(readme, load_json):
    text = _read(ROOT / readme).replace("\u2212", "-")
    w = load_json("14_wgi_measurement.json")
    s = w["summary"]
    assert f"{s['n_distinguishable_2020_2024']} de {s['n_changes_2020_2024']}" in text or f"{s['n_distinguishable_2020_2024']} of {s['n_changes_2020_2024']}" in text
    assert f"{w['reliability_worst_case']['index']['reliability']:.2f}" in text
    assert f"{s['mean_noise_to_within_sd']:.2f}" in text
    g = w["eq1_eq2_by_content"]["violence_related"]["EQ1"]
    assert f"{g['coef']:.3f}" in text
    c = load_json("15_growth_ceiling_robustness.json")
    lo = c["frequentist_loco"]["0.90"]
    assert f"{lo['n_same_sign_and_p05']} de {lo['n_fits']}" in text or f"{lo['n_same_sign_and_p05']} of {lo['n_fits']}" in text
    b = c["bayesian_without_excluded"]["0.90"]
    assert f"{b['beta_mean']:.2f}" in text and f"{b['beta_hdi_lo']:.2f}" in text


def test_pipeline_readme_lists_every_module():
    text = _read(PIPE / "README.md")
    mods = sorted(p.name for p in PIPE.glob("[0-9][0-9]_*.py"))
    assert len(mods) == 15
    for m in mods:
        assert f"`{m}`" in text, f"{m} missing from econometric_pipeline/pipeline/README.md"
    assert "percentile ranks, 0" not in text                         # the WGI 0-100 score is not a percentile rank


def test_checkpoint_banner_matches_json(load_json):
    text = _read(ROOT / "documentation" / "checkpoint.md").replace("\u2212", "-")
    assert "STATUS:BEGIN" in text and "no refleja los resultados vigentes" in text
    fe, md = load_json("02_fe_results.json"), load_json("04_mediation.json")
    for x in (fe["EQ1"]["coef_key_cl"], fe["EQ2"]["coef_key_cl"], md["indirect_full"]):
        assert f"{abs(x):.3f}" in text
    assert f"{fe['EQ1']['pval_key_cl']:.3f}" in text and f"{md['p_boot']:.3f}" in text


def test_main_readme_code_table_lists_the_gdl_scripts():
    text = _read(ROOT / "README.md")
    for s in ("01_build_global_panel.py", "03_baselines.py", "04_gnn.py", "05_verdict.py"):
        assert s in text
