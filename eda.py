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
plt.title("Average governance indicators")
plt.show()