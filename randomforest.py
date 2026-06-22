import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import shap
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("panel_ready_for_modeling.csv")

print("Shape:", df.shape)

# =========================
# TARGET (CRECIMIENTO ECONÓMICO)
# =========================

TARGET = "gdp_growth"

# =========================
# FEATURES
# =========================

features = [
    "control_corruption",
    "political_stability",
    "rule_of_law",
    "homicide_rate_log",
    "exports_percent_gdp",
    "fdi_percent_gdp",
    "gdp_per_capita_log",
    "inflation",
    "population_log",
    "tourist_arrivals_log",
    "unemployment"
]

# eliminar filas incompletas SOLO en target
df = df.dropna(subset=[TARGET])

X = df[features]
y = df[TARGET]

# =========================
# TRAIN / TEST SPLIT (IMPORTANTE: temporal simple)
# =========================

df = df.sort_values(["country_code", "year"])

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

X_train = train[features]
y_train = train[TARGET]

X_test = test[features]
y_test = test[TARGET]

# =========================
# RANDOM FOREST MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# PREDICTION + EVALUATION
# =========================

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n=== MODEL PERFORMANCE ===")
print("R2:", round(r2, 4))
print("RMSE:", round(rmse, 4))

# =========================
# FEATURE IMPORTANCE (BASELINE)
# =========================

importances = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importances)

# =========================
# SHAP (INTERPRETABILIDAD REAL)
# =========================

print("\nCalculando SHAP...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# resumen global
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

# =========================
# SHAP IMPORTANCE NUMÉRICO
# =========================

shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": features,
    "shap_importance": shap_importance
}).sort_values("shap_importance", ascending=False)

print("\n=== SHAP IMPORTANCE ===")
print(shap_df)

# =========================
# EXPORT RESULTS
# =========================

importances.to_csv("rf_feature_importance.csv", index=False)
shap_df.to_csv("shap_feature_importance.csv", index=False)

print("\nDONE:")
print("- rf_feature_importance.csv")
print("- shap_feature_importance.csv")
print("- shap_summary.png")