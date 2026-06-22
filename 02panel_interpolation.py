import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# LOAD DATASET
# =========================

df_raw = pd.read_csv("panel_ready_for_modeling.csv")

print("=== ORIGINAL SHAPE ===")
print(df_raw.shape)

# =========================
# GUARDA METADATA (CRÍTICO)
# =========================

meta = df_raw[["country_code", "country_name", "year"]].copy()

# copia de trabajo separada
df = df_raw.copy()

# =========================
# ORDER PANEL
# =========================

df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

# =========================
# VARIABLES NUMÉRICAS
# =========================

features = [
    "control_corruption",
    "political_stability",
    "rule_of_law",
    "homicide_rate_log",
    "exports_percent_gdp",
    "fdi_percent_gdp",
    "gdp_growth",
    "gdp_per_capita_log",
    "inflation",
    "population_log",
    "tourist_arrivals_log",
    "unemployment"
]

# asegurar que existen
features = [f for f in features if f in df.columns]

# =========================
# INTERPOLACIÓN POR PAÍS
# =========================

def interpolate_country(group):
    group = group.sort_values("year").copy()

    group[features] = group[features].interpolate(method="linear")
    group[features] = group[features].ffill().bfill()

    return group

df = df.groupby("country_code", group_keys=False).apply(interpolate_country)

df = df.reset_index(drop=True)

# =========================
# CHECK MISSING
# =========================

print("\n=== MISSING AFTER INTERPOLATION ===")
print(df[features].isna().sum())

# =========================
# FEATURE MATRIX
# =========================

X = df[features].to_numpy()

# =========================
# STANDARDIZATION
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== FEATURES USADAS EN PCA ===")
print(features)

missing = [f for f in features if f not in df.columns]
print("Missing features:", missing)

# =========================
# PCA FULL
# =========================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_

print("\n=== PCA VARIANCE EXPLAINED ===")
for i, v in enumerate(explained[:10]):
    print(f"PC{i+1}: {v:.4f}")

# =========================
# SELECT COMPONENTS
# =========================

cumulative = np.cumsum(explained)
n_components = np.argmax(cumulative >= 0.90) + 1

print(f"\nComponentes para 90% varianza: {n_components}")

pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(X_scaled)

# =========================
# FINAL DATAFRAME (FIX CLAVE)
# =========================

pca_cols = [f"PC{i+1}" for i in range(n_components)]
df_pca = pd.DataFrame(X_pca_final, columns=pca_cols)

df_final = pd.concat([
    meta.reset_index(drop=True),
    df_pca.reset_index(drop=True)
], axis=1)

# =========================
# SAVE
# =========================

df_final.to_csv("panel_pca_ready.csv", index=False)

print("\n=== FINAL SHAPE ===")
print(df_final.shape)

print("\n=== DONE ===")
print("Saved: panel_pca_ready.csv")