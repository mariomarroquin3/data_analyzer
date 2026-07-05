import pandas as pd
import numpy as np

# =========================================================
# 1. CARGA DEL DATASET FINAL
# =========================================================

df = pd.read_csv("final_research_dataset.csv")

print("=== DATASET ORIGINAL ===")
print(df.shape)
print(df.columns)

# =========================================================
# 2. LIMPIEZA BÁSICA DE PANEL
# =========================================================

# asegurar orden panel
df = df.sort_values(["country_code", "year"])

# eliminar duplicados por seguridad
df = df.drop_duplicates(["country_code", "year"])

# asegurar tipos
df["year"] = df["year"].astype(int)

# =========================================================
# 3. TRANSFORMACIONES LOG (ESENCIALES)
# =========================================================

# GDP per capita
df["gdp_per_capita_log"] = np.log(df["gdp_per_capita"])

# población
df["population_log"] = np.log(df["population"])

# turismo (maneja ceros)
df["tourist_arrivals_log"] = np.log1p(df["tourist_arrivals"])

# =========================================================
# 4. OUTLIERS (HOMICIDIOS)
# =========================================================

# winsorization 1% - 99%
lower = df["homicide_rate"].quantile(0.01)
upper = df["homicide_rate"].quantile(0.99)

df["homicide_rate_winsor"] = df["homicide_rate"].clip(lower, upper)

# versión log (útil para modelos)
df["homicide_rate_log"] = np.log1p(df["homicide_rate_winsor"])

# =========================================================
# 5. CREACIÓN DE TREND TEMPORAL
# =========================================================

df["year_trend"] = df["year"] - 2000

# =========================================================
# 6. VERIFICACIÓN DE CONSISTENCIA PANEL
# =========================================================

print("\n=== PANEL CHECK ===")
print("Shape:", df.shape)

print("\nDuplicados (debe ser 0):")
print(df.duplicated(["country_code", "year"]).sum())

print("\nCobertura por país:")
print(df.groupby("country_name")["year"].agg(["min", "max", "count"]))

# =========================================================
# 7. PREPARACIÓN PARA PCA (SOLO WGI)
# =========================================================

wgi_cols = [
    "control_corruption",
    "political_stability",
    "rule_of_law"
]

print("\n=== WGI SUMMARY ===")
print(df[wgi_cols].describe())

print("\nCorrelación WGI:")
print(df[wgi_cols].corr())

# =========================================================
# 8. DATASET FINAL PARA MODELADO
# =========================================================

model_cols = [
    "country_code",
    "country_name",
    "year",
    "year_trend",

    # WGI (crudo por ahora)
    "control_corruption",
    "political_stability",
    "rule_of_law",

    # crimen
    "homicide_rate_log",

    # economía
    "exports_percent_gdp",
    "fdi_percent_gdp",
    "gdp_growth",
    "gdp_per_capita_log",
    "inflation",
    "population_log",
    "tourist_arrivals_log",
    "unemployment"
]

df_model = df[model_cols].copy()

# =========================================================
# 9. VALIDACIÓN FINAL
# =========================================================

print("\n=== FINAL DATASET ===")
print(df_model.shape)

print("\nMissing values:")
print(df_model.isna().sum()[df_model.isna().sum() > 0])

print("\nEjemplo:")
print(df_model.head())

# =========================================================
# 10. EXPORTAR
# =========================================================

df_model.to_csv("panel_ready_for_modeling.csv", index=False)

print("\n=== EXPORT COMPLETADO ===")
print("Archivo: panel_ready_for_modeling.csv")