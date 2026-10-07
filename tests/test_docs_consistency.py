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
