import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "econometric_pipeline" / "pipeline"
JSON_DIR = PIPE / "json"

EXPECTED_COUNTRIES = {"COL", "CRI", "DOM", "ECU", "GTM", "HND", "MEX", "NIC", "PAN", "PER", "SLV"}
WGI_COLS = [
    "rule_of_law", "control_corruption", "political_stability",
    "voice_accountability", "government_effectiveness", "regulatory_quality",
]


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return pd.read_csv(PIPE / "panel_enriched.csv", encoding="utf-8")


@pytest.fixture(scope="session")
def load_json():
    def _load(name: str) -> dict:
        with open(JSON_DIR / name, encoding="utf-8") as f:
            return json.load(f)
    return _load
