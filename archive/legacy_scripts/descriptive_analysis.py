# ============================================================
# ELSALVADOR_DESCRIPTIVE.PY
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_research_dataset.csv")

df = df.sort_values(["country_code", "year"])

# ============================================================
# 2. EL SALVADOR ONLY
# ============================================================

df = df[df["country_code"] == "SLV"].copy()

print("\n")
print("=" * 70)
print("EL SALVADOR DESCRIPTIVE ANALYSIS")
print("=" * 70)

# ============================================================
# 3. PCA INSTITUTIONAL INDEX
# ============================================================

inst_vars = [
    "rule_of_law",
    "control_corruption",
    "political_stability"
]

X = df[inst_vars].dropna()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=1)

df.loc[X.index, "institution_index"] = pca.fit_transform(X_std)

if pca.components_[0][0] < 0:
    df["institution_index"] *= -1

# ============================================================
# 4. DESCRIPTIVE STATS
# ============================================================

print("\nDESCRIPTIVE STATISTICS")
print("-" * 70)

desc_vars = [
    "homicide_rate",
    "rule_of_law",
    "political_stability",
    "control_corruption",
    "institution_index",
    "fdi_percent_gdp",
    "gdp_growth"
]

print(df[desc_vars].describe().round(3))

# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

print("\nCORRELATION MATRIX")
print("-" * 70)

corr_vars = [
    "homicide_rate",
    "institution_index",
    "fdi_percent_gdp",
    "gdp_growth"
]

corr_matrix = df[corr_vars].corr()

print(corr_matrix.round(3))

# ============================================================
# 6. KEY CORRELATIONS
# ============================================================

print("\nKEY CORRELATIONS")
print("-" * 70)

pairs = {
    "Homicides vs Institutions":
        df["homicide_rate"].corr(df["institution_index"]),

    "Homicides vs Growth":
        df["homicide_rate"].corr(df["gdp_growth"]),

    "Institutions vs Growth":
        df["institution_index"].corr(df["gdp_growth"]),

    "Institutions vs FDI":
        df["institution_index"].corr(df["fdi_percent_gdp"]),

    "FDI vs Growth":
        df["fdi_percent_gdp"].corr(df["gdp_growth"])
}

for k, v in pairs.items():
    print(f"{k:<30}: {v:.3f}")

# ============================================================
# 7. LAGGED RELATIONSHIPS
# ============================================================

print("\nLAGGED CORRELATIONS")
print("-" * 70)

df["institution_t1"] = df["institution_index"].shift(-1)
df["growth_t1"] = df["gdp_growth"].shift(-1)

lag1 = df["homicide_rate"].corr(df["institution_t1"])

lag2 = df["institution_index"].corr(df["growth_t1"])

print(f"Homicides(t) -> Institutions(t+1): {lag1:.3f}")
print(f"Institutions(t) -> Growth(t+1):    {lag2:.3f}")

# ============================================================
# 8. HOMICIDE PEAK
# ============================================================

print("\nHOMICIDE SHOCK")
print("-" * 70)

peak_idx = df["homicide_rate"].idxmax()

peak_year = df.loc[peak_idx, "year"]
peak_value = df.loc[peak_idx, "homicide_rate"]

last_year = df["year"].max()

last_value = df.loc[
    df["year"] == last_year,
    "homicide_rate"
].iloc[0]

reduction = (
    (peak_value - last_value)
    / peak_value
) * 100

print(f"Peak year:           {peak_year}")
print(f"Peak homicide rate:  {peak_value:.2f}")

print(f"Latest year:         {last_year}")
print(f"Latest homicide:     {last_value:.2f}")

print(f"Reduction (%):       {reduction:.2f}")

# ============================================================
# 9. INSTITUTIONAL CHANGE
# ============================================================

print("\nINSTITUTIONAL CHANGE")
print("-" * 70)

for var in [
    "rule_of_law",
    "political_stability",
    "control_corruption"
]:

    start = df[var].iloc[0]
    end = df[var].iloc[-1]

    change = end - start

    print(f"{var:<25}: {change:.2f}")

# ============================================================
# 10. GDP CHANGE
# ============================================================

print("\nGDP GROWTH EXTREMES")
print("-" * 70)

gdp_max_idx = df["gdp_growth"].idxmax()
gdp_min_idx = df["gdp_growth"].idxmin()

print(
    "Max GDP growth:",
    df.loc[gdp_max_idx, "year"],
    round(df.loc[gdp_max_idx, "gdp_growth"], 2)
)

print(
    "Min GDP growth:",
    df.loc[gdp_min_idx, "year"],
    round(df.loc[gdp_min_idx, "gdp_growth"], 2)
)

# ============================================================
# 11. EXPORT TABLES
# ============================================================

corr_matrix.to_csv(
    "elsalvador_correlation_matrix.csv"
)

df[desc_vars].describe().to_csv(
    "elsalvador_descriptive_stats.csv"
)

print("\nTables exported successfully.")

# ============================================================
# 12. RAW TIME SERIES
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 9),
    sharex=True
)

axes[0].plot(df["year"], df["homicide_rate"])
axes[0].set_title("Homicide Rate")

axes[1].plot(df["year"], df["institution_index"])
axes[1].set_title("Institutional Index (PCA)")

axes[2].plot(df["year"], df["fdi_percent_gdp"])
axes[2].set_title("FDI (% GDP)")

axes[3].plot(df["year"], df["gdp_growth"])
axes[3].set_title("GDP Growth")

plt.tight_layout()
plt.show()