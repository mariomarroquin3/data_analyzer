import pandas as pd
import numpy as np

INPUT_FILE="final_research_dataset.csv"
MASTER_FILE="panel_master_dataset.csv"
MODEL_FILE="panel_ready_for_modeling.csv"

print("="*70)
print("BUILDING MASTER PANEL DATASET")
print("="*70)

df=pd.read_csv(INPUT_FILE)

required=["country_code","country_name","year","population","gdp_per_capita","tourist_arrivals","homicide_rate","exports_percent_gdp","fdi_percent_gdp","gdp_growth","inflation","unemployment","control_corruption","political_stability","rule_of_law"]
missing=[c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df=df.sort_values(["country_code","year"]).drop_duplicates(["country_code","year"]).reset_index(drop=True)
df["year"]=df["year"].astype(int)

safe=lambda s: np.where(s>0,np.log(s),np.nan)
df["population_log"]=safe(df["population"])
df["gdp_per_capita_log"]=safe(df["gdp_per_capita"])
df["tourist_arrivals_log"]=np.log1p(df["tourist_arrivals"])

# NOTE (fixed): this used to winsorize homicide_rate at the 1st/99th
# percentile before taking log1p. With only 179 obs, the 4 clipped points
# were ALL El Salvador -- its 2015/2016 violence peak (107.6 -> 83.6,
# 85.1 -> 83.6) AND its 2023/2024 post-security-reform lows (2.24 -> 5.43,
# 1.90 -> 5.43). That is exactly the abrupt security shift this whole
# research project studies, silently flattened before it ever reaches
# EQ1, undocumented anywhere in the paper or the active pipeline. Use the
# raw rate directly; log1p already handles the right-skew and any zeros.
df["homicide_rate_log"]=np.log1p(df["homicide_rate"])

if "imports_percent_gdp" in df.columns:
    df["trade_percent_gdp"]=df["exports_percent_gdp"]+df["imports_percent_gdp"]
else:
    df["trade_percent_gdp"]=np.nan

df["time_trend"]=df["year"]-df["year"].min()

for lag in [1,2,3]:
    df[f"homicide_rate_lag{lag}"]=df.groupby("country_code")["homicide_rate"].shift(lag)
    df[f"homicide_rate_log_lag{lag}"]=df.groupby("country_code")["homicide_rate_log"].shift(lag)

df.to_csv(MASTER_FILE,index=False)

cols=["country_code","country_name","year","time_trend","rule_of_law","control_corruption","political_stability","homicide_rate","homicide_rate_log","homicide_rate_lag1","homicide_rate_lag2","homicide_rate_lag3","homicide_rate_log_lag1","homicide_rate_log_lag2","homicide_rate_log_lag3","population","population_log","gdp_per_capita","gdp_per_capita_log","gdp_growth","inflation","unemployment","exports_percent_gdp","trade_percent_gdp","fdi_percent_gdp","tourist_arrivals","tourist_arrivals_log"]
df[[c for c in cols if c in df.columns]].to_csv(MODEL_FILE,index=False)
print("Done")
