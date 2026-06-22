import pandas as pd

# =========================
# CARGAR DATASET
# =========================

df = pd.read_csv("WGI1.csv")

# =========================
# IDENTIFICADORES
# =========================

id_vars = ["Country Name", "Country Code", "Series Name", "Series Code"]

# =========================
# WIDE → LONG
# =========================

df_long = df.melt(
    id_vars=id_vars,
    var_name="year",
    value_name="value"
)

# =========================
# LIMPIAR AÑO
# =========================

df_long["year"] = (
    df_long["year"]
    .str.extract(r"(\d{4})")
    .astype(int)
)

# =========================
# FILTRAR AÑOS
# =========================

df_long = df_long[(df_long["year"] >= 2000) & (df_long["year"] <= 2024)]

# =========================
# PIVOT A PANEL
# =========================

df_panel = df_long.pivot_table(
    index=["Country Code", "Country Name", "year"],
    columns="Series Code",
    values="value"
).reset_index()

# =========================
# RENOMBRAR VARIABLES
# =========================

df_panel = df_panel.rename(columns={
    "Country Code": "country_code",
    "Country Name": "country",
    "GOV_WGI_RL.SC": "rule_of_law",
    "GOV_WGI_PV.SC": "political_stability",
    "GOV_WGI_CC.SC": "control_corruption"
})

# =========================
# ORDENAR
# =========================

df_panel = df_panel.sort_values(["country", "year"])

# =========================
# CHECK
# =========================

print("\n=== DATASET FINAL WGI ===")
print(df_panel.head())

print("\n=== MISSING VALUES ===")
print(df_panel.isna().sum())

print("\n=== COBERTURA ===")
print(df_panel.groupby("country")["year"].agg(["min", "max", "count"]))

# =========================
# GUARDAR
# =========================

df_panel.to_csv("wgi_clean_panel.csv", index=False)

print("\nGUARDADO: wgi_clean_panel.csv")