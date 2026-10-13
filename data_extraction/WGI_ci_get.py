"""
WGI_ci_get.py
Download the uncertainty information that accompanies the WGI governance scores
used in the panel: for each of the 18 countries, six dimensions and years
2000-2024, the 0-100 governance score, the lower/upper bounds of its 90%
confidence interval, and the number of data sources behind the estimate.

World Bank API, source 3 (Worldwide Governance Indicators, 2025 revision):
    GOV_WGI_<DIM>.SC     governance score (0-100)
    GOV_WGI_<DIM>.SC_LB  lower bound of the 90% confidence interval
    GOV_WGI_<DIM>.SC_UB  upper bound of the 90% confidence interval
    GOV_WGI_<DIM>.SR     number of sources

Output: foundational_datasets/wgi_confidence_intervals.csv (long by country,
dimension and year). Module 14 reads this file, so the pipeline itself needs no
network access; re-run this script only to refresh it.

Reference: Jiang, Jain, Romano & Boyce (2025), "The Worldwide Governance
Indicators: Revised Methodology for Measuring Governance Using Perception
Data", World Bank (the 90% interval is the estimate +/- 1.64 conditional
standard deviations, mapped linearly to the 0-100 scale).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

COUNTRIES = ["BHS", "BLZ", "BRA", "CHL", "COL", "CRI", "DOM", "ECU", "GTM", "HND",
             "HTI", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY"]
DIMS = {"RL": "rule_of_law", "PV": "political_stability", "CC": "control_corruption",
        "VA": "voice_accountability", "GE": "government_effectiveness", "RQ": "regulatory_quality"}
SUFFIX = {"SC": "score", "SC_LB": "lb", "SC_UB": "ub", "SR": "n_sources"}
OUT = Path(__file__).resolve().parent.parent / "foundational_datasets" / "wgi_confidence_intervals.csv"


def fetch(job, tries=6):
    """GET with retries: a silently dropped series would bias the diagnostics."""
    c, d, s = job
    url = f"https://api.worldbank.org/v2/country/{c}/indicator/GOV_WGI_{d}.{s}?format=json&per_page=200&date=2000:2024"
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                js = r.json()
                rows = js[1] if len(js) > 1 and js[1] else []
                return [(c, DIMS[d], int(o["date"]), SUFFIX[s], o["value"]) for o in rows if o["value"] is not None]
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} attempts for {job}: {last}")


if __name__ == "__main__":
    jobs = [(c, d, s) for c in COUNTRIES for d in DIMS for s in SUFFIX]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for chunk in ex.map(fetch, jobs) for r in chunk]
    long = pd.DataFrame(rows, columns=["country_code", "dim", "year", "stat", "value"])
    wide = long.pivot_table(index=["country_code", "dim", "year"], columns="stat", values="value").reset_index()
    wide = wide[["country_code", "dim", "year", "score", "lb", "ub", "n_sources"]].sort_values(["country_code", "dim", "year"])
    assert len(wide) == len(COUNTRIES) * len(DIMS) * 24, f"unexpected size {len(wide)} (2001 is missing in the WGI)"
    OUT.parent.mkdir(exist_ok=True)
    wide.to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(wide)} rows)")
