import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv("final_research_dataset.csv")

# =========================
# FILTRO (sin 2020)
# =========================
plot_df = df[df["year"] != 2020].copy()

x = plot_df["homicide_rate"].values
y = plot_df["gdp_growth"].values

# =========================
# REGRESIÓN LINEAL SIMPLE (SOLO VISUAL)
# =========================
coef = np.polyfit(x, y, 1)

# Predicción para R²
y_pred = coef[0] * x + coef[1]
r2 = r2_score(y, y_pred)

x_line = np.linspace(x.min(), x.max(), 200)
y_line = coef[0] * x_line + coef[1]

# =========================
# PLOT
# =========================
plt.figure(figsize=(9,6))

for country in plot_df["country_name"].unique():
    sub = plot_df[plot_df["country_name"] == country]

    plt.scatter(
        sub["homicide_rate"],
        sub["gdp_growth"],
        alpha=0.7,
        s=40,
        label=country
    )

# Línea de tendencia
plt.plot(
    x_line,
    y_line,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Linear OLS trend (visual)"
)

# =========================
# R² en la figura
# =========================
plt.text(
    0.02, 0.95,
    f"$R^2 = {r2:.3f}$",
    transform=plt.gca().transAxes,
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

# =========================
# FORMATO FINAL
# =========================
plt.xlabel("Homicidios (por 100,000 habitantes)")
plt.ylabel("Crecimiento PIB (%)")
plt.title("Crimen vs Crecimiento (excluyendo 2020)")
plt.grid(alpha=0.3)

plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()