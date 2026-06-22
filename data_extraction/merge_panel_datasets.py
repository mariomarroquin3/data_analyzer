import pandas as pd
import numpy as np

# ============================================================================
# 1. MAPEO ISO2 -> ISO3 para Centro América y Caribe
# ============================================================================
ISO2_TO_ISO3 = {
    'SV': 'SLV',  # El Salvador
    'GT': 'GTM',  # Guatemala
    'HN': 'HND',  # Honduras
    'NI': 'NIC',  # Nicaragua
    'CR': 'CRI',  # Costa Rica
    'PA': 'PAN',  # Panamá
    'DO': 'DOM',  # República Dominicana
    'CO': 'COL',  # Colombia
}

# ============================================================================
# 2. DICCIONARIO ISO3 -> NOMBRE EN ESPAÑOL (consistente)
# ============================================================================
ISO3_TO_SPANISH_NAME = {
    'COL': 'Colombia',
    'SLV': 'El Salvador',
    'GTM': 'Guatemala',
    'HND': 'Honduras',
    'NIC': 'Nicaragua',
    'CRI': 'Costa Rica',
    'PAN': 'Panamá',
    'DOM': 'República Dominicana',
}

# ============================================================================
# 3. FUNCIONES DE LIMPIEZA
# ============================================================================

def normalize_text(value):
    """Strip espacios y uppercase"""
    if pd.isna(value):
        return value
    return str(value).strip().upper()

def normalize_year(value):
    """Convertir year a int"""
    try:
        return int(value)
    except:
        return np.nan

def convert_iso2_to_iso3(code):
    """Convierte ISO2 a ISO3 si está en el mapeo"""
    if pd.isna(code):
        return code
    code_clean = str(code).strip().upper()
    return ISO2_TO_ISO3.get(code_clean, code_clean)

# ============================================================================
# 4. CARGAR Y LIMPIAR DATASETS
# ============================================================================

print("=" * 80)
print("CARGANDO Y LIMPIANDO DATASETS")
print("=" * 80)

# Dataset 1: WGI (ya está en ISO3)
print("\n[1] Cargando WGI dataset (ISO3)...")
wgi = pd.read_csv('wgi_clean_panel.csv')
print(f"   Shape inicial: {wgi.shape}")

wgi['country_code'] = wgi['country_code'].apply(normalize_text)
wgi['year'] = wgi['year'].apply(normalize_year)
wgi = wgi[['country_code', 'year', 'control_corruption', 'political_stability', 'rule_of_law']]
wgi = wgi.dropna(subset=['country_code', 'year'])
print(f"   Shape después de limpieza: {wgi.shape}")
print(f"   Códigos únicos: {sorted(wgi['country_code'].unique())}")

# Dataset 2: Homicides (ISO2 -> convertir a ISO3)
print("\n[2] Cargando Homicides dataset (ISO2 -> ISO3)...")
homicides = pd.read_csv('security_wgi_homicides.csv')
print(f"   Shape inicial: {homicides.shape}")

homicides['country_code'] = homicides['country_code'].apply(normalize_text)
homicides['country_code'] = homicides['country_code'].apply(convert_iso2_to_iso3)
homicides['year'] = homicides['year'].apply(normalize_year)
homicides = homicides[['country_code', 'year', 'homicide_rate']]
homicides = homicides.dropna(subset=['country_code', 'year'])
print(f"   Shape después de limpieza: {homicides.shape}")
print(f"   Códigos únicos: {sorted(homicides['country_code'].unique())}")

# Dataset 3: Economic (ISO2 -> convertir a ISO3)
print("\n[3] Cargando Economic dataset (ISO2 -> ISO3)...")
economic = pd.read_csv('dataset_centroamerica_panel_v2.csv')
print(f"   Shape inicial: {economic.shape}")

economic['country_code'] = economic['country_code'].apply(normalize_text)
economic['country_code'] = economic['country_code'].apply(convert_iso2_to_iso3)
economic['year'] = economic['year'].apply(normalize_year)
economic = economic[[
    'country_code', 'year', 'exports_percent_gdp', 'fdi_percent_gdp',
    'gdp_growth', 'gdp_per_capita', 'inflation', 'population',
    'tourist_arrivals', 'unemployment'
]]
economic = economic.dropna(subset=['country_code', 'year'])
print(f"   Shape después de limpieza: {economic.shape}")
print(f"   Códigos únicos: {sorted(economic['country_code'].unique())}")

# ============================================================================
# 5. VALIDAR INTERSECCIÓN DE KEYS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDACIÓN DE INTERSECCIÓN")
print("=" * 80)

# Keys (country_code, year)
wgi_keys = set(zip(wgi['country_code'], wgi['year']))
homicides_keys = set(zip(homicides['country_code'], homicides['year']))
economic_keys = set(zip(economic['country_code'], economic['year']))

print(f"\nWGI keys:       {len(wgi_keys)}")
print(f"Homicides keys: {len(homicides_keys)}")
print(f"Economic keys:  {len(economic_keys)}")

# Intersecciones
wgi_h_intersection = wgi_keys & homicides_keys
wgi_e_intersection = wgi_keys & economic_keys
h_e_intersection = homicides_keys & economic_keys
three_way = wgi_keys & homicides_keys & economic_keys

print(f"\nWGI ∩ Homicides: {len(wgi_h_intersection)} registros")
print(f"WGI ∩ Economic:  {len(wgi_e_intersection)} registros")
print(f"Homicides ∩ Economic: {len(h_e_intersection)} registros")
print(f"WGI ∩ Homicides ∩ Economic: {len(three_way)} registros (FINAL)")

# Diagnosticar qué falta
if len(three_way) == 0:
    print("\n⚠️ ADVERTENCIA: No hay intersección de los tres datasets!")
    print("\nAnálisis de keys faltantes:")
    
    # Países únicos por dataset
    wgi_countries = set([k[0] for k in wgi_keys])
    hom_countries = set([k[0] for k in homicides_keys])
    econ_countries = set([k[0] for k in economic_keys])
    
    print(f"\n  Países en WGI: {sorted(wgi_countries)}")
    print(f"  Países en Homicides: {sorted(hom_countries)}")
    print(f"  Países en Economic: {sorted(econ_countries)}")
    
    print(f"\n  En WGI pero NO en Homicides: {sorted(wgi_countries - hom_countries)}")
    print(f"  En WGI pero NO en Economic: {sorted(wgi_countries - econ_countries)}")
    print(f"  En Homicides pero NO en WGI: {sorted(hom_countries - wgi_countries)}")
    print(f"  En Economic pero NO en WGI: {sorted(econ_countries - wgi_countries)}")
else:
    print("\n✓ Intersección válida encontrada")

# ============================================================================
# 6. MERGE SECUENCIAL POR country_code Y year
# ============================================================================

print("\n" + "=" * 80)
print("EJECUTANDO MERGES")
print("=" * 80)

# Primero: WGI + Homicides
print("\n[Paso 1] Merge WGI + Homicides...")
merged = wgi.merge(
    homicides,
    on=['country_code', 'year'],
    how='inner'
)
print(f"   Resultado: {merged.shape[0]} registros")

# Segundo: resultado + Economic
print("[Paso 2] Merge resultado + Economic...")
merged = merged.merge(
    economic,
    on=['country_code', 'year'],
    how='inner'
)
print(f"   Resultado: {merged.shape[0]} registros")

# ============================================================================
# 7. ASIGNAR NOMBRES DE PAÍSES EN ESPAÑOL
# ============================================================================

print("\n[Paso 3] Asignando nombres en español...")
merged['country_name'] = merged['country_code'].map(ISO3_TO_SPANISH_NAME)

# Verificar si quedaron NA (country_code no estaba en el diccionario)
na_names = merged['country_name'].isna().sum()
if na_names > 0:
    print(f"   ⚠️ {na_names} registros con country_name NA (códigos no reconocidos)")
    print(f"   Códigos sin mapeo: {merged[merged['country_name'].isna()]['country_code'].unique()}")
    # Limpiar estos registros
    merged = merged.dropna(subset=['country_name'])
    print(f"   Registros después de limpieza: {merged.shape[0]}")
else:
    print(f"   ✓ Todos los nombres asignados correctamente")

# ============================================================================
# 8. REORDENAR COLUMNAS SEGÚN ESPECIFICACIÓN
# ============================================================================

final_columns = [
    'country_code',
    'country_name',
    'year',
    'control_corruption',
    'political_stability',
    'rule_of_law',
    'homicide_rate',
    'exports_percent_gdp',
    'fdi_percent_gdp',
    'gdp_growth',
    'gdp_per_capita',
    'inflation',
    'population',
    'tourist_arrivals',
    'unemployment'
]

final_dataset = merged[final_columns].sort_values(['country_code', 'year']).reset_index(drop=True)

# ============================================================================
# 9. REPORTE FINAL
# ============================================================================

print("\n" + "=" * 80)
print("REPORTE FINAL")
print("=" * 80)

print(f"\nShape del dataset final: {final_dataset.shape}")
print(f"Registros por país:")
country_counts = final_dataset['country_name'].value_counts().sort_index()
for country, count in country_counts.items():
    print(f"  {country}: {count}")

print(f"\nRango de años: {final_dataset['year'].min()} - {final_dataset['year'].max()}")
print(f"Años únicos: {sorted(final_dataset['year'].unique())}")

print(f"\nMissing values:")
missing = final_dataset.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, count in missing.items():
        print(f"  {col}: {count}")
else:
    print("  ✓ Sin valores faltantes")

print(f"\n{final_dataset.head(10).to_string()}")

# ============================================================================
# 10. GUARDAR DATASET FINAL
# ============================================================================

print("\n" + "=" * 80)
print("GUARDANDO DATASET")
print("=" * 80)

final_dataset.to_csv('final_research_dataset.csv', index=False)
print("\n✓ Dataset guardado en: final_research_dataset.csv")
print(f"\nÚltimos 5 registros:")
print(final_dataset.tail().to_string())
