import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("panel_ready_for_modeling.csv")

df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

print("Shape:", df.shape)

# =========================
# VARIABLES
# =========================

# STEP A: violencia → instituciones
X_A = df[["homicide_rate_log", "control_corruption"]]
y_A = df["rule_of_law"]

# STEP B: instituciones → inversión/turismo
X_B = df[["rule_of_law", "control_corruption"]]
y_B = df["fdi_percent_gdp"]

# agregamos turismo como segundo output clave
y_B2 = df["tourist_arrivals_log"].copy()

# imputación simple SOLO para evitar NaN en el modelo
y_B2 = y_B2.interpolate().ffill().bfill()

# STEP C: economía final
X_C = df[[
    "fdi_percent_gdp",
    "tourist_arrivals_log",
    "rule_of_law",
    "unemployment",
    "inflation",
    "gdp_per_capita_log",
    "exports_percent_gdp"
]]

y_C = df["gdp_growth"]

# =========================
# MODELO A
# =========================

model_A = LinearRegression()
model_A.fit(X_A, y_A)
pred_A = model_A.predict(X_A)

print("\n=== MODEL A: Homicides → Rule of Law ===")
print("R2:", r2_score(y_A, pred_A))
print("Coef:", dict(zip(X_A.columns, model_A.coef_)))

df["rule_of_law_hat"] = pred_A

# =========================
# MODELO B (FDI)
# =========================

model_B = LinearRegression()
model_B.fit(df[["rule_of_law_hat", "control_corruption"]], y_B)
pred_B = model_B.predict(df[["rule_of_law_hat", "control_corruption"]])

print("\n=== MODEL B1: Institutions → FDI ===")
print("R2:", r2_score(y_B, pred_B))
print("Coef:", dict(zip(["rule_of_law_hat", "control_corruption"], model_B.coef_)))

df["fdi_hat"] = pred_B

# =========================
# MODELO B2 (Tourism)
# =========================

model_B2 = LinearRegression()
model_B2.fit(df[["rule_of_law_hat"]], y_B2)
pred_B2 = model_B2.predict(df[["rule_of_law_hat"]])

print("\n=== MODEL B2: Institutions → Tourism ===")
print("R2:", r2_score(y_B2, pred_B2))
print("Coef:", model_B2.coef_)

df["tourism_hat"] = pred_B2

# =========================
# MODELO C (GDP Growth)
# =========================

X_C_final = df[[
    "fdi_hat",
    "tourism_hat",
    "rule_of_law_hat",
    "unemployment",
    "inflation",
    "gdp_per_capita_log",
    "exports_percent_gdp"
]]

model_C = LinearRegression()
model_C.fit(X_C_final, y_C)
pred_C = model_C.predict(X_C_final)

print("\n=== MODEL C: Full Growth Equation ===")
print("R2:", r2_score(y_C, pred_C))
print("Coef:", dict(zip(X_C_final.columns, model_C.coef_)))

# =========================
# SUMMARY TABLE
# =========================

df_results = df[[
    "country_code",
    "country_name",
    "year",
    "gdp_growth"
]].copy()

df_results["gdp_pred"] = pred_C

df_results.to_csv("chain_model_results.csv", index=False)

print("\n=== DONE ===")
print("Saved: chain_model_results.csv")