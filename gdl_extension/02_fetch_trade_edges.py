"""
02_fetch_trade_edges.py
════════════════════════════════════════════════════════════════════════
GDL extension, step 2 -- bilateral trade edges from the CEPII Gravity database.

Trade is the natural graph for an economic GNN (Sellami et al., 2024; Araujo et
al., 2026 use bilateral trade). To avoid information leakage, edge weights are
averaged over a FIXED EARLY WINDOW (2000-2008) that ends before every
forecast-evaluation year of the models, so the graph is static and carries no
information from the years being predicted.

Source: CEPII Gravity (V202211), `tradeflow_baci` (exports of the origin to
the destination, thousand current US$; BACI is CEPII's harmonised version of
UN Comtrade). The ~200 MB zip is downloaded once to gdl_extension/data_raw/
(git-ignored).

Output: gdl_extension/data/edges_trade.csv  (iso_o, iso_d, flow_kusd,
share_of_exports_o, share_of_imports_d) for pairs among the economies in
countries.csv.
════════════════════════════════════════════════════════════════════════
"""
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
RAW.mkdir(exist_ok=True)
ZIP = RAW / "Gravity_csv_V202211.zip"
URL = "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"
WINDOW = (2000, 2008)

if not ZIP.exists() or ZIP.stat().st_size < 150e6:
    print("downloading", URL)
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = ZIP.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(ZIP)
print(f"zip: {ZIP.stat().st_size / 1e6:.0f} MB")

iso = set(pd.read_csv(HERE / "data" / "countries.csv")["iso3"])
with zipfile.ZipFile(ZIP) as zf:
    name = next(n for n in zf.namelist() if n.lower().endswith(".csv") and "gravity" in n.lower())
    print("reading", name)
    head = pd.read_csv(zf.open(name), nrows=0).columns.tolist()
    flow = "tradeflow_baci" if "tradeflow_baci" in head else "tradeflow_comtrade_o"
    print("trade column:", flow)
    parts = []
    for chunk in pd.read_csv(zf.open(name), usecols=["year", "iso3_o", "iso3_d", flow], chunksize=500_000):
        chunk = chunk[chunk["year"].between(*WINDOW) & chunk["iso3_o"].isin(iso) & chunk["iso3_d"].isin(iso) & (chunk["iso3_o"] != chunk["iso3_d"])]
        parts.append(chunk)
df = pd.concat(parts)
df[flow] = pd.to_numeric(df[flow], errors="coerce")
pair = df.groupby(["iso3_o", "iso3_d"])[flow].mean().reset_index().rename(columns={"iso3_o": "iso_o", "iso3_d": "iso_d", flow: "flow_kusd"})
pair = pair[pair["flow_kusd"] > 0]
pair["share_of_exports_o"] = pair["flow_kusd"] / pair.groupby("iso_o")["flow_kusd"].transform("sum")
pair["share_of_imports_d"] = pair["flow_kusd"] / pair.groupby("iso_d")["flow_kusd"].transform("sum")
pair.to_csv(HERE / "data" / "edges_trade.csv", index=False)
print(f"{len(pair)} directed trade pairs among {pair['iso_o'].nunique()} exporters and {pair['iso_d'].nunique()} importers; "
      f"window {WINDOW[0]}-{WINDOW[1]}")
print(pair.sort_values("flow_kusd", ascending=False).head(5).to_string(index=False))
