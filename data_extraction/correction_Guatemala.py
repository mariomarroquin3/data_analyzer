import requests
import pandas as pd

# ======================================
# 1. CARGAR DATASET ACTUAL
# ======================================

archivo_original = "dataset_centroamerica_panel.csv"

df_panel = pd.read_csv(archivo_original)

print("Dataset cargado")
print(f"Filas: {len(df_panel)}")
print(f"Missing GDP per capita antes: {df_panel['gdp_per_capita'].isna().sum()}")

# ======================================
# 2. DESCARGAR GDP PER CAPITA GUATEMALA
# ======================================

url = (
    "https://api.worldbank.org/v2/country/GT/"
    "indicator/NY.GDP.PCAP.KD"
    "?format=json&per_page=100"
)

response = requests.get(url, timeout=30)

if response.status_code != 200:
    raise Exception(f"Error HTTP {response.status_code}")

resultado = response.json()

if len(resultado) < 2:
    raise Exception("No se recibieron datos válidos")

guatemala_data = []

for obs in resultado[1]:

    valor = obs["value"]
    fecha = obs["date"]

    if valor is None:
        continue

    if not fecha.isdigit():
        continue

    year = int(fecha)

    if 2000 <= year <= 2024:

        guatemala_data.append({
            "country_code": "GT",
            "year": year,
            "gdp_per_capita_new": valor
        })

df_gt = pd.DataFrame(guatemala_data)

print("\nSerie descargada")
print(df_gt.head())

# ======================================
# 3. MERGE
# ======================================

df_panel = df_panel.merge(
    df_gt,
    on=["country_code", "year"],
    how="left"
)

# ======================================
# 4. RELLENAR SOLO LOS NaN
# ======================================

df_panel["gdp_per_capita"] = (
    df_panel["gdp_per_capita"]
    .fillna(df_panel["gdp_per_capita_new"])
)

# ======================================
# 5. LIMPIAR COLUMNA AUXILIAR
# ======================================

df_panel.drop(
    columns=["gdp_per_capita_new"],
    inplace=True
)

# ======================================
# 6. VERIFICACIÓN
# ======================================

missing_final = df_panel["gdp_per_capita"].isna().sum()

print("\n=== RESULTADO ===")
print(f"Missing GDP per capita después: {missing_final}")

print("\nGuatemala (primeros años):")
print(
    df_panel.loc[
        df_panel["country"] == "Guatemala",
        ["year", "gdp_per_capita"]
    ]
    .sort_values("year")
    .head()
)

print("\nGuatemala (últimos años):")
print(
    df_panel.loc[
        df_panel["country"] == "Guatemala",
        ["year", "gdp_per_capita"]
    ]
    .sort_values("year")
    .tail()
)

# ======================================
# 7. GUARDAR NUEVA VERSIÓN
# ======================================

archivo_salida = "dataset_centroamerica_panel_v2.csv"

df_panel.to_csv(
    archivo_salida,
    index=False
)

print(f"\nDataset guardado: {archivo_salida}")

# ======================================
# 8. RESUMEN GENERAL
# ======================================

print("\n=== MISSING VALUES FINALES ===")
print(df_panel.isna().sum())