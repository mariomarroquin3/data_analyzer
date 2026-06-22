import requests
import pandas as pd

# =========================
# PAÍSES
# =========================

PAISES = {
    'SV': 'El Salvador',
    'GT': 'Guatemala',
    'HN': 'Honduras',
    'NI': 'Nicaragua',
    'CR': 'Costa Rica',
    'PA': 'Panamá',
    'DO': 'República Dominicana',
    'CO': 'Colombia'
}

AÑOS = set(range(2000, 2025))

# =========================
# INDICADORES
# =========================

INDICADORES = {
    "RL.EST": "rule_of_law",
    "PV.EST": "political_stability",
    "CC.EST": "control_corruption",
    "VC.IHR.PSRC.P5": "homicide_rate"
}

# =========================
# DESCARGA LONG FORMAT
# =========================

data = []

for code, country in PAISES.items():

    for ind, name in INDICADORES.items():

        url = f"https://api.worldbank.org/v2/country/{code}/indicator/{ind}?format=json&per_page=20000"

        try:
            r = requests.get(url, timeout=30)

            if r.status_code != 200:
                print(f"Error {r.status_code} en {country} - {ind}")
                continue

            js = r.json()

            if len(js) < 2:
                continue

            for obs in js[1]:

                year = obs.get("date")
                value = obs.get("value")

                if year is None or not year.isdigit():
                    continue

                year = int(year)

                if year in AÑOS:

                    data.append({
                        "country_code": code,
                        "country": country,
                        "year": year,
                        "indicator": name,
                        "value": value
                    })

        except Exception as e:
            print(f"Error {country}-{ind}: {e}")

# =========================
# DATAFRAME LONG
# =========================

df = pd.DataFrame(data)

print("\n=== LONG DATASET ===")
print(df.head())

# =========================
# PIVOT (DATASET FINAL)
# =========================

df_final = df.pivot_table(
    index=["country_code", "country", "year"],
    columns="indicator",
    values="value"
).reset_index()

# =========================
# ORDENAR
# =========================

df_final = df_final.sort_values(["country", "year"])

# =========================
# CHECK GENERAL
# =========================

print("\n=== DATASET FINAL ===")
print(df_final.head())

print("\n=== MISSING VALUES ===")
print(df_final.isna().sum())

print("\n=== COBERTURA POR PAÍS ===")
print(
    df_final.groupby("country")["year"]
    .agg(["min", "max", "count"])
)

# =========================
# VALIDACIÓN ESPECÍFICA
# =========================

print("\n=== HOMICIDIOS POR PAÍS ===")
if "homicide_rate" in df_final.columns:
    print(
        df_final.groupby("country")["homicide_rate"]
        .count()
        .sort_values(ascending=False)
    )
else:
    print("⚠ homicide_rate no se generó correctamente")

# =========================
# GUARDAR
# =========================

df_final.to_csv("security_wgi_homicides.csv", index=False)

print("\n=== DATASET GUARDADO ===")
print("security_wgi_homicides.csv")