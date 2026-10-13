"""
01_build_global_panel.py
════════════════════════════════════════════════════════════════════════
GDL extension, step 1 -- a GLOBAL country-year panel and static country graphs.

The 18-country panel of the main pipeline is far too small for a graph neural
network. This script builds the larger data set a geometric-deep-learning
analysis needs, from public sources only:

  nodes / features  World Bank API (WGI 2025 revision governance scores and
                    90% interval bounds; WDI controls; homicide rate), ~215
                    economies, 2000-2024
  node metadata     World Bank country list (region, income level, capital
                    coordinates)
  edges (static)    land borders (mledoze/countries), CEPII GeoDist
                    (population-weighted distance, contiguity, common official
                    language, colonial links)

Output (gdl_extension/data/): global_panel.csv, countries.csv,
edges_static.csv, coverage.json.  Bilateral trade edges are built separately
(02_fetch_trade_edges.py) because the CEPII gravity file is large.

Nothing here is modelled; the coverage report is printed so that the sample
rule (see README.md) can be fixed BEFORE any model is fitted.
════════════════════════════════════════════════════════════════════════
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
YEARS = (2000, 2024)

WGI_DIMS = {"RL": "rule_of_law", "PV": "political_stability", "CC": "control_corruption",
            "VA": "voice_accountability", "GE": "government_effectiveness", "RQ": "regulatory_quality"}
WDI = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "fdi_percent_gdp": "BX.KLT.DINV.WD.GD.ZS",
    "unemployment": "SL.UEM.TOTL.ZS",
    "inflation": "FP.CPI.TOTL.ZG",
    "exports_percent_gdp": "NE.EXP.GNFS.ZS",
    "population": "SP.POP.TOTL",
    "homicide_rate": "VC.IHR.PSRC.P5",
}
# WGI series: score, lower and upper bound of the 90% interval
INDICATORS = dict(WDI)
for k, name in WGI_DIMS.items():
    INDICATORS[name] = f"GOV_WGI_{k}.SC"
    INDICATORS[name + "_lb"] = f"GOV_WGI_{k}.SC_LB"
    INDICATORS[name + "_ub"] = f"GOV_WGI_{k}.SC_UB"

SESSION = requests.Session()


def get_json(url, tries=6):
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:                       # transient failures must not drop a series silently
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} attempts: {url}: {last}")


def fetch_indicator(item):
    name, code = item
    rows, page = [], 1
    while True:
        js = get_json(f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=1000"
                      f"&date={YEARS[0]}:{YEARS[1]}&page={page}")
        meta, data = js[0], (js[1] or [])
        rows += [(o["countryiso3code"], int(o["date"]), o["value"]) for o in data
                 if o.get("value") is not None and o.get("countryiso3code")]
        if page >= int(meta["pages"]):
            break
        page += 1
    return name, rows


print("== World Bank country list")
js = get_json("https://api.worldbank.org/v2/country?format=json&per_page=400")
countries = pd.DataFrame([{
    "iso3": c["id"], "name": c["name"], "region": c["region"]["value"].strip(), "income": c["incomeLevel"]["value"].strip(),
    "lon": float(c["longitude"]) if c["longitude"] else np.nan, "lat": float(c["latitude"]) if c["latitude"] else np.nan}
    for c in js[1] if c["region"]["id"] != "NA"])
print(f"   {len(countries)} economies (aggregates removed); regions: {countries['region'].nunique()}")

print(f"== Downloading {len(INDICATORS)} indicators for all economies")
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch_indicator, INDICATORS.items()))
long = pd.concat([pd.DataFrame(rows, columns=["iso3", "year", "value"]).assign(var=name) for name, rows in results])
wide = long.pivot_table(index=["iso3", "year"], columns="var", values="value").reset_index()
wide = wide[wide["iso3"].isin(countries["iso3"])]
wide["log_gdp_pc"] = np.log(wide["gdp_per_capita"])
wide["log_population"] = np.log(wide["population"])
wide["log_homicide"] = np.log(wide["homicide_rate"].clip(lower=0.1))     # same floor idea as the main pipeline's log transform
print(f"   panel: {len(wide)} country-years, {wide['iso3'].nunique()} economies")

print("== Static edges")
mz = requests.get("https://raw.githubusercontent.com/mledoze/countries/master/countries.json", timeout=60).json()
border = {c["cca3"]: set(c.get("borders", [])) for c in mz}
lang = {c["cca3"]: set(c.get("languages", {})) for c in mz}
iso = set(countries["iso3"])
rows = []
for a in sorted(iso):
    for b in sorted(border.get(a, set()) & iso):
        rows.append((a, b))
borders = pd.DataFrame(rows, columns=["iso_o", "iso_d"]).assign(border=1)
print(f"   land-border pairs (directed): {len(borders)}")

# CEPII GeoDist: distances and cultural links. CEPII uses a few non-WB ISO codes.
geo = pd.read_stata("http://www.cepii.fr/distance/dist_cepii.dta")
FIX = {"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "YUG": "SRB", "PAL": "PSE"}
for c in ("iso_o", "iso_d"):
    geo[c] = geo[c].replace(FIX)
geo = geo[geo["iso_o"].isin(iso) & geo["iso_d"].isin(iso) & (geo["iso_o"] != geo["iso_d"])]
geo = geo[["iso_o", "iso_d", "dist", "distw", "contig", "comlang_off", "comlang_ethno", "colony", "comcol", "curcol", "col45"]]
edges = geo.merge(borders, on=["iso_o", "iso_d"], how="left")
edges["border"] = edges["border"].fillna(0).astype(int)
print(f"   CEPII pairs: {len(edges)} | covering {edges['iso_o'].nunique()} economies | border flag agrees with CEPII contig on "
      f"{(edges['border'] == edges['contig']).mean():.1%} of pairs")

print("== Coverage (reported BEFORE any modelling)")
core = ["gdp_growth", "log_gdp_pc", "fdi_percent_gdp", "inflation", "exports_percent_gdp", "log_population"]
wgi = list(WGI_DIMS.values())
panel = wide[wide["year"].between(*YEARS)].copy()
cov = panel.groupby("iso3").agg(**{f"n_{v}": (v, "count") for v in core + ["unemployment", "log_homicide"] + wgi})
cov["n_core_min"] = cov[[f"n_{v}" for v in core]].min(axis=1)
cov["n_wgi_min"] = cov[[f"n_{v}" for v in wgi]].min(axis=1)
n_years = YEARS[1] - YEARS[0] + 1
summary = {
    "n_economies_any_data": int(len(cov)),
    "n_with_core_20plus": int((cov["n_core_min"] >= 20).sum()),
    "n_with_core_20plus_and_wgi_20plus": int(((cov["n_core_min"] >= 20) & (cov["n_wgi_min"] >= 20)).sum()),
    "n_with_core_wgi_and_homicide_15plus": int(((cov["n_core_min"] >= 20) & (cov["n_wgi_min"] >= 20) & (cov["n_log_homicide"] >= 15)).sum()),
    "n_with_core_wgi_homicide_unemp_15plus": int(((cov["n_core_min"] >= 20) & (cov["n_wgi_min"] >= 20) & (cov["n_log_homicide"] >= 15) & (cov["n_unemployment"] >= 15)).sum()),
    "years": list(YEARS), "n_years": n_years,
}
for k, v in summary.items():
    print(f"   {k}: {v}")
reg = countries.set_index("iso3")["region"]
elig = cov[(cov["n_core_min"] >= 20) & (cov["n_wgi_min"] >= 20) & (cov["n_log_homicide"] >= 15)].index
print("   eligible by region:", reg.loc[elig].value_counts().to_dict())

panel.to_csv(OUT / "global_panel.csv", index=False)
countries.to_csv(OUT / "countries.csv", index=False)
edges.to_csv(OUT / "edges_static.csv", index=False)
cov.reset_index().to_csv(OUT / "coverage_by_country.csv", index=False)
json.dump(summary, open(OUT / "coverage.json", "w"), indent=2)
print(f"saved to {OUT}")
