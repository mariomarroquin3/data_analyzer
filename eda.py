import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("final_research_dataset.csv")

print("Shape:", df.shape)
print("\nColumnas:", df.columns.tolist())

print("\nRango de años:")
print(df["year"].min(), "-", df["year"].max())

print("\nPaíses:")
print(df["country_name"].unique())



print("\nObservaciones por país:")
print(df.groupby("country_name").size())

print("\nCobertura temporal por país:")
print(df.groupby("country_name")["year"].agg(["min","max","count"]))

desc = df.describe().T
print(desc)
print("\nSkewness:")
print(df.skew(numeric_only=True).sort_values(ascending=False))

corr = df.corr(numeric_only=True)

print(corr["homicide_rate"].sort_values(ascending=False))
print("\nCorrelación con GDP growth:")
print(corr["gdp_growth"].sort_values(ascending=False))


plt.scatter(df["homicide_rate"], df["gdp_growth"])
plt.xlabel("Homicide rate")
plt.ylabel("GDP growth")
plt.title("Crime vs Economic Growth")
plt.show()

for c in df["country_name"].unique():
    subset = df[df["country_name"] == c]
    plt.plot(subset["year"], subset["homicide_rate"], label=c)

plt.legend()
plt.title("Homicide trends by country")
plt.show()

df.groupby("country_name")[[
    "rule_of_law",
    "control_corruption",
    "political_stability"
]].mean().plot(kind="bar")
plt.title("Indicadores de gobernanza promedio")
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv("final_research_dataset.csv")

# =========================
# EXCLUIR 2020 (outlier COVID)
# =========================
plot_df = df[df["year"] != 2020].copy()

# =========================
# FIGURA PRINCIPAL
# =========================
plt.figure(figsize=(9, 6))

# Scatter por país (colores automáticos)
for country in sorted(plot_df["country_name"].unique()):
    subset = plot_df[plot_df["country_name"] == country]

    plt.scatter(
        subset["homicide_rate"],
        subset["gdp_growth"],
        label=country,
        s=40,
        alpha=0.75
    )

# =========================
# LÍNEA DE TENDENCIA GLOBAL
# =========================
x = plot_df["homicide_rate"].values
y = plot_df["gdp_growth"].values

coef = np.polyfit(x, y, 1)  # regresión lineal simple
x_line = np.linspace(x.min(), x.max(), 200)
y_line = coef[0] * x_line + coef[1]

plt.plot(
    x_line,
    y_line,
    color="black",
    linewidth=2,
    linestyle="--",
    label="Linear trend"
)

# =========================
# FORMATO FINAL
# =========================
plt.xlabel("Homicidios (por 100,000 habitantes)")
plt.ylabel("Crecimiento GDP(%)")
plt.title("Crimen vs Crecimiento (excluyendo 2020)")

plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()

# Exportación alta calidad (IMPORTANTE para paper)
plt.savefig("crime_vs_growth.png", dpi=300, bbox_inches="tight")

plt.show()