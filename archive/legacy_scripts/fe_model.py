import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ==========================================================
# FIXED EFFECTS PANEL MODEL
# Violencia -> Instituciones -> FDI / Turismo -> Crecimiento
# ==========================================================

print("=" * 70)
print("        FIXED EFFECTS PANEL MODEL (CLUSTER ROBUST)")
print("=" * 70)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("panel_ready_for_modeling.csv")

df = (
    df
    .sort_values(["country_code", "year"])
    .reset_index(drop=True)
)

df["country_code"] = df["country_code"].astype("category")
df["year"] = df["year"].astype("category")

# ==========================================================
# OPTIONAL TRANSFORMATIONS
# ==========================================================

# evitar problemas numéricos

for col in [
    "exports_percent_gdp",
    "inflation",
    "gdp_per_capita_log",
    "unemployment"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# GENERIC ESTIMATOR
# ==========================================================

def estimate_fe(formula, dataframe, fitted_name=None):

    model = smf.ols(
        formula=formula,
        data=dataframe,
        missing="drop"
    )

    used = model.data.frame.copy()

    results = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": used["country_code"]
        }
    )

    print(results.summary())

    if fitted_name is not None:
        dataframe.loc[used.index, fitted_name] = results.fittedvalues

    return results


# ==========================================================
# MODEL A
# Violence -> Rule of Law
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL A")
print("Violence → Rule of Law")
print("=" * 70)

formula_A = """
rule_of_law
~
homicide_rate_log
+ C(country_code)
+ C(year)
"""

model_A = estimate_fe(
    formula_A,
    df,
    "rule_of_law_hat"
)

# ==========================================================
# MODEL B1
# Rule of Law -> FDI
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL B1")
print("Rule of Law → FDI")
print("=" * 70)

formula_B1 = """
fdi_percent_gdp
~
rule_of_law_hat
+ C(country_code)
+ C(year)
"""

model_B1 = estimate_fe(
    formula_B1,
    df,
    "fdi_hat"
)

# ==========================================================
# MODEL B2
# Rule of Law -> Tourism
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL B2")
print("Rule of Law → Tourism")
print("=" * 70)

formula_B2 = """
tourist_arrivals_log
~
rule_of_law_hat
+ C(country_code)
+ C(year)
"""

model_B2 = estimate_fe(
    formula_B2,
    df,
    "tourism_hat"
)

# ==========================================================
# MODEL C
# Growth Equation
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL C")
print("FDI + Tourism + Institutions → Growth")
print("=" * 70)

formula_C = """
gdp_growth
~
fdi_hat
+ tourism_hat
+ rule_of_law_hat
+ exports_percent_gdp
+ inflation
+ unemployment
+ C(country_code)
+ C(year)
"""

model_C = estimate_fe(
    formula_C,
    df
)

# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Model":[
        "Violence → Rule of Law",
        "Rule of Law → FDI",
        "Rule of Law → Tourism",
        "Growth Equation"
    ],

    "Observations":[
        int(model_A.nobs),
        int(model_B1.nobs),
        int(model_B2.nobs),
        int(model_C.nobs)
    ],

    "R2":[
        model_A.rsquared,
        model_B1.rsquared,
        model_B2.rsquared,
        model_C.rsquared
    ],

    "Adj_R2":[
        model_A.rsquared_adj,
        model_B1.rsquared_adj,
        model_B2.rsquared_adj,
        model_C.rsquared_adj
    ],

    "AIC":[
        model_A.aic,
        model_B1.aic,
        model_B2.aic,
        model_C.aic
    ],

    "BIC":[
        model_A.bic,
        model_B1.bic,
        model_B2.bic,
        model_C.bic
    ]

})

summary.to_csv(
    "fe_chain_results.csv",
    index=False
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)
print(summary)