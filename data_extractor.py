import requests
import pandas as pd
import time

# =========================
# CONFIGURACIÓN
# =========================

PAISES = ['SV', 'GT', 'HN', 'NI', 'CR', 'PA', 'DO', 'CO']

PAISES_NOMBRES = {
    'SV': 'El Salvador',
    'GT': 'Guatemala',
    'HN': 'Honduras',
    'NI': 'Nicaragua',
    'CR': 'Costa Rica',
    'PA': 'Panamá',
    'DO': 'República Dominicana',
    'CO': 'Colombia'
}

INDICADORES = {
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'fdi_percent_gdp': 'BX.KLT.DINV.WD.GD.ZS',
    'unemployment': 'SL.UEM.TOTL.ZS',
    'inflation': 'FP.CPI.TOTL.ZG',
    'exports_percent_gdp': 'NE.EXP.GNFS.ZS',
    'tourist_arrivals': 'ST.INT.ARVL',
    'population': 'SP.POP.TOTL'
}

AÑOS = set(range(2000, 2025))

# =========================
# EXTRACCIÓN (FORMATO LONG)
# =========================

data = []

print("=== DESCARGANDO DATOS WORLD BANK ===")

for country in PAISES:
    country_name = PAISES_NOMBRES[country]
    print(f"\nPaís: {country_name}")

    for var_name, var_code in INDICADORES.items():

        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{var_code}?format=json&per_page=20000"

        try:
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                print(f"  ❌ Error HTTP {response.status_code} en {var_name}")
                continue

            result = response.json()

            if len(result) < 2:
                print(f"  ⚠️ Sin datos para {var_name}")
                continue

            for obs in result[1]:
                year = obs["date"]
                value = obs["value"]

                if year is None or not year.isdigit():
                    continue

                year = int(year)

                if year in AÑOS:
                    data.append({
                        "country_code": country,
                        "country": country_name,
                        "year": year,
                        "variable": var_name,
                        "value": value
                    })

        except Exception as e:
            print(f"  ⚠️ Error en {var_name}: {e}")

        time.sleep(0.1)

# =========================
# DATAFRAME LONG
# =========================

df_long = pd.DataFrame(data)

print("\n=== DATA LONG CREADO ===")
print(df_long.head())

# =========================
# PIVOT A FORMATO PANEL (WIDE)
# =========================

df_panel = df_long.pivot_table(
    index=["country_code", "country", "year"],
    columns="variable",
    values="value"
).reset_index()

# ordenar
df_panel = df_panel.sort_values(["country", "year"])

# =========================
# GUARDAR
# =========================

output_file = "dataset_centroamerica_panel.csv"
df_panel.to_csv(output_file, index=False)

# =========================
# RESUMEN
# =========================

print("\n=== DATASET FINAL ===")
print(f"Archivo: {output_file}")
print(f"Filas: {len(df_panel)}")
print(f"Países: {df_panel['country'].nunique()}")
print(f"Años: {df_panel['year'].nunique()}")

print("\n=== MISSING VALUES ===")
print(df_panel.isnull().sum())