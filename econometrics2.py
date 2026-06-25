# ============================================================
# 0. LIBRERÍAS
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD + ORDER
# ============================================================

df = pd.read_csv("panel_ready_for_modeling.csv")
df = df.sort_values(["country_code", "year"])

country_code = "SLV"

# ============================================================
# 2. FILTER EL SALVADOR
# ============================================================

df = df[df["country_code"] == country_code].copy()

# ============================================================
# 3. PCA (INSTITUTIONS) - SAFE VERSION
# ============================================================

inst_vars = [
    "rule_of_law",
    "control_corruption",
    "political_stability"
]

X = df[inst_vars].dropna()

if X.shape[0] < 3:
    raise ValueError(f"Not enough data for PCA: {X.shape}")

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=1)
df.loc[X.index, "institution_index"] = pca.fit_transform(X_std)

# asegurar interpretación positiva
if pca.components_[0][0] < 0:
    df["institution_index"] *= -1

# ============================================================
# 4. LAGS (COHERENTES)
# ============================================================

df["homicide_lag1"] = df["homicide_rate_log"].shift(1)
df["fdi_lag1"] = df["fdi_percent_gdp"].shift(1)

# ============================================================
# 5. CLEAN FINAL DATASET
# ============================================================

df_model = df.dropna(subset=[
    "homicide_lag1",
    "institution_index",
    "fdi_percent_gdp",
    "gdp_growth"
])

# ============================================================
# 6. MODELO 1: VIOLENCE → INSTITUTIONS
# ============================================================

m1 = smf.ols("""
    institution_index ~ homicide_lag1 + year
""", data=df_model).fit(cov_type="HC1")

print("\n=== MODEL 1: Violence → Institutions ===")
print(m1.summary())

# ============================================================
# 7. MODELO 2: INSTITUTIONS → FDI
# ============================================================

m2 = smf.ols("""
    fdi_percent_gdp ~ institution_index + year
""", data=df_model).fit(cov_type="HC1")

print("\n=== MODEL 2: Institutions → FDI ===")
print(m2.summary())

# ============================================================
# 8. MODELO 3: FDI → GROWTH
# ============================================================

m3 = smf.ols("""
    gdp_growth ~ fdi_percent_gdp + institution_index + year
""", data=df_model).fit(cov_type="HC1")

print("\n=== MODEL 3: Growth Equation ===")
print(m3.summary())

# ============================================================
# 9. RESULTADOS RESUMIDOS
# ============================================================

results_table = pd.DataFrame({
    "Model": ["Violence→Inst", "Inst→FDI", "FDI→Growth"],
    "Coef": [
        m1.params["homicide_lag1"],
        m2.params["institution_index"],
        m3.params["fdi_percent_gdp"]
    ],
    "P-value": [
        m1.pvalues["homicide_lag1"],
        m2.pvalues["institution_index"],
        m3.pvalues["fdi_percent_gdp"]
    ]
})

print("\n=== SUMMARY TABLE ===")
print(results_table)

# ============================================================
# 10. VISUALIZACIÓN SIMPLE (SUBPLOTS - RECOMENDADO)
# ============================================================

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

axes[0].plot(df_model["year"], df_model["homicide_rate_log"])
axes[0].set_title("Violence (Homicides)")

axes[1].plot(df_model["year"], df_model["institution_index"])
axes[1].set_title("Institutions (PCA)")

axes[2].plot(df_model["year"], df_model["fdi_percent_gdp"])
axes[2].set_title("FDI (% GDP)")

axes[3].plot(df_model["year"], df_model["gdp_growth"])
axes[3].set_title("GDP Growth")

plt.tight_layout()
plt.show()

# ============================================================
# 11. VISUALIZACIÓN NORMALIZADA (OPCIONAL, PAPER STYLE)
# ============================================================

viz = df_model.copy()

cols = [
    "homicide_rate_log",
    "institution_index",
    "fdi_percent_gdp",
    "gdp_growth"
]

scaler = MinMaxScaler()
viz[cols] = scaler.fit_transform(viz[cols])

plt.figure()

plt.plot(viz["year"], viz["homicide_rate_log"], label="Violence (scaled)")
plt.plot(viz["year"], viz["institution_index"], label="Institutions (scaled)")
plt.plot(viz["year"], viz["fdi_percent_gdp"], label="FDI (scaled)")
plt.plot(viz["year"], viz["gdp_growth"], label="Growth (scaled)")

plt.title("Mechanism: Violence → Institutions → Economy (Scaled)")
plt.xlabel("Year")
plt.legend()
plt.grid(True)

plt.show()