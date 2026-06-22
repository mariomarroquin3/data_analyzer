import pandas as pd

df = pd.read_csv("final_research_dataset.csv")

print("Observaciones totales:", len(df))


print("\nObservaciones por país:")
print(df.groupby("country_name").size())

print("\nAños por país:")
print(df.groupby("country_name")["year"].agg(["min","max","count"]))

print("\nMissing values por variable:")
print(df.isna().mean().sort_values(ascending=False))