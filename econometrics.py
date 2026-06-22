import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# LOAD
# =========================
df = pd.read_csv("panel_ready_for_modeling.csv")

df = df.sort_values(["country_code", "year"])

# =========================
# FIX NA (CRÍTICO)
# =========================
df = df.dropna()

# =========================
# CONTROLES BASE
# =========================
controls = [
    "control_corruption",
    "political_stability",
    "exports_percent_gdp",
    "inflation",
    "gdp_per_capita_log",
    "population_log",
    "unemployment"
]

# =========================
# A: Violence → Institutions
# =========================
X_A = df[["homicide_rate_log"] + controls]
X_A = sm.add_constant(X_A)
y_A = df["rule_of_law"]

model_A = sm.OLS(y_A, X_A).fit()
df["rule_of_law_hat"] = model_A.predict(X_A)

print("\n=== MODEL A ===")
print(model_A.summary())

# =========================
# B1: Institutions → FDI
# =========================
X_B1 = df[["rule_of_law_hat", "control_corruption", "political_stability"]]
X_B1 = sm.add_constant(X_B1)
y_B1 = df["fdi_percent_gdp"]

model_B1 = sm.OLS(y_B1, X_B1).fit()
df["fdi_hat"] = model_B1.predict(X_B1)

print("\n=== MODEL B1 ===")
print(model_B1.summary())

# =========================
# B2: Institutions → Tourism
# =========================
X_B2 = df[["rule_of_law_hat", "control_corruption", "political_stability"]]
X_B2 = sm.add_constant(X_B2)
y_B2 = df["tourist_arrivals_log"]

model_B2 = sm.OLS(y_B2, X_B2).fit()
df["tourism_hat"] = model_B2.predict(X_B2)

print("\n=== MODEL B2 ===")
print(model_B2.summary())

# =========================
# C: Growth Equation
# =========================
X_C = df[
    [
        "fdi_hat",
        "tourism_hat",
        "rule_of_law_hat",
        "exports_percent_gdp",
        "inflation",
        "gdp_per_capita_log",
        "unemployment"
    ]
]

X_C = sm.add_constant(X_C)
y_C = df["gdp_growth"]

model_C = sm.OLS(y_C, X_C).fit()

print("\n=== MODEL C ===")
print(model_C.summary())

# =========================
# SAVE RESULTS
# =========================
df.to_csv("chain_model_final.csv", index=False)