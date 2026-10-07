import requests
import pandas as pd
import time

# =========================
# CONFIGURACIÓN
# =========================

# Original 11 countries + 7 added October 2026 under an ex-ante coverage rule:
# homicide rate >= 15 of 25 years, each WGI dimension >= 20 of 24, each core WDI
# series >= 20 of 25, unemployment >= 15 (all other LAC economies fail it because
# WDI does not publish a core series, e.g. exports for JAM/TTO, inflation for ARG/VEN).
PAISES = ['SV', 'GT', 'HN', 'NI', 'CR', 'PA', 'DO', 'CO', 'MX', 'EC', 'PE',
          'BS', 'BZ', 'BR', 'CL', 'HT', 'PY', 'UY']

PAISES_NOMBRES = {
    'SV': 'El Salvador',
    'GT': 'Guatemala',
    'HN': 'Honduras',
    'NI': 'Nicaragua',
    'CR': 'Costa Rica',
    'PA': 'Panamá',
    'DO': 'República Dominicana',
    'CO': 'Colombia',
    'MX': 'México',
    'EC': 'Ecuador',
    'PE': 'Perú',
    'BS': 'Bahamas',
    'BZ': 'Belice',
    'BR': 'Brasil',
    'CL': 'Chile',
    'HT': 'Haití',
    'PY': 'Paraguay',
    'UY': 'Uruguay',
}

INDICADORES = {
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'fdi_percent_gdp': 'BX.KLT.DINV.WD.GD.ZS',
    'unemployment': 'SL.UEM.TOTL.ZS',
    'inflation': 'FP.CPI.TOTL.ZG',
    'exports_percent_gdp': 'NE.EXP.GNFS.ZS',
    'imports_percent_gdp': 'NE.IMP.GNFS.ZS',
    'tourist_arrivals': 'ST.INT.ARVL',
    'population': 'SP.POP.TOTL',
    # Added September 2026: remittances are a first-order channel in this
    # region (El Salvador alone runs ~20-25% of GDP), plausibly linked to
    # both violence (emigration push) and growth (household consumption) --
    # confirmed live via direct API call before adoption.
    'remittances_percent_gdp': 'BX.TRF.PWKR.DT.GD.ZS',
}

AÑOS = set(range(2000, 2025))

# =========================
# EXTRACCIÓN (FORMATO LONG)
# =========================


def fetch(url, tries=6):
    """GET with retries: transient API timeouts/resets must not silently drop a
    country-indicator series (an earlier run lost COL/NIC imports and HTI
    unemployment this way)."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} attempts: {last}")

data = []

print("=== DESCARGANDO DATOS WORLD BANK ===")

for country in PAISES:
    country_name = PAISES_NOMBRES[country]
    print(f"\nPaís: {country_name}")

    for var_name, var_code in INDICADORES.items():

        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{var_code}?format=json&per_page=200&date=2000:2024"

        try:
            response = fetch(url)

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