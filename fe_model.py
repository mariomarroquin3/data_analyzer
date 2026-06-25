import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("panel_ready_for_modeling.csv")

df = df.sort_values(["country_code", "year"])

# =========================
# PANEL VARIABLES
# =========================

# FE encoding
df["country_code"] = df["country_code"].astype("category")
df["year"] = df["year"].astype(int)

# =========================
# MODEL A: Violence → Institutions
# =========================

model_A = smf.ols(
    formula="""
    rule_of_law ~ homicide_rate_log
                + control_corruption
                + political_stability
                + C(country_code)
                + C(year)
    """,
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["country_code"]})

print("\n=== MODEL A (FE) ===")
print(model_A.summary())

# predicted institutions
df["rule_of_law_hat"] = model_A.fittedvalues

# =========================
# MODEL B1: Institutions → FDI
# =========================

model_B1 = smf.ols(
    formula="""
    fdi_percent_gdp ~ rule_of_law_hat
                    + control_corruption
                    + political_stability
                    + C(country_code)
                    + C(year)
    """,
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["country_code"]})

print("\n=== MODEL B1 (FE) ===")
print(model_B1.summary())

df["fdi_hat"] = model_B1.fittedvalues

# =========================
# MODEL B2: Institutions → Tourism
# =========================

model_B2 = smf.ols(
    formula="""
    tourist_arrivals_log ~ rule_of_law_hat
                         + control_corruption
                         + political_stability
                         + C(country_code)
                         + C(year)
    """,
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["country_code"]})

print("\n=== MODEL B2 (FE) ===")
print(model_B2.summary())

df["tourism_hat"] = model_B2.fittedvalues

# =========================
# MODEL C: Growth equation (final stage)
# =========================

model_C = smf.ols(
    formula="""
    gdp_growth ~ fdi_hat
              + tourism_hat
              + rule_of_law_hat
              + exports_percent_gdp
              + inflation
              + gdp_per_capita_log
              + unemployment
              + C(country_code)
              + C(year)
    """,
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["country_code"]})

print("\n=== MODEL C (FE) ===")
print(model_C.summary())

# =========================
# SAVE RESULTS
# =========================

results = pd.DataFrame({
    "model": ["A", "B1", "B2", "C"],
    "r2": [
        model_A.rsquared,
        model_B1.rsquared,
        model_B2.rsquared,
        model_C.rsquared
    ]
})

results.to_csv("fe_chain_results.csv", index=False)

print("\nDONE → fe_chain_results.csv")