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

# Dataset 1+2: WGI + Homicides (ISO2 -> ISO3)
#
# NOTE (fixed): these used to be loaded from two different sources --
# 'wgi_clean_panel.csv' (institutions, via WGI_test.py, which itself
# depends on a manually-downloaded 'WGI1.csv' that was never committed
# to the repo, so its exact query/vintage could not be verified or
# rerun) and 'security_wgi_homicides.csv' (homicides only, via
# WGI_get.py, which ALSO pulled institutional indicators but under the
# wrong World Bank codes -- ".EST", the governance ESTIMATE scale
# (~ -2.5 to 2.5), instead of the ".PER.RNK" percentile-rank scale
# (0-100) that every downstream module in this pipeline documents and
# expects). WGI_get.py's indicator codes are now fixed, so its output
# is the single, live, reproducible source for both institutions and
# homicides.
print("\n[1] Cargando WGI + Homicides dataset (ISO2 -> ISO3)...")
wgi_hom = pd.read_csv('security_wgi_homicides.csv')
print(f"   Shape inicial: {wgi_hom.shape}")

wgi_hom['country_code'] = wgi_hom['country_code'].apply(normalize_text)
wgi_hom['country_code'] = wgi_hom['country_code'].apply(convert_iso2_to_iso3)
wgi_hom['year'] = wgi_hom['year'].apply(normalize_year)
wgi_hom = wgi_hom.dropna(subset=['country_code', 'year'])

wgi = wgi_hom[['country_code', 'year', 'control_corruption', 'political_stability', 'rule_of_law']]
print(f"   WGI shape después de limpieza: {wgi.shape}")
print(f"   Códigos únicos: {sorted(wgi['country_code'].unique())}")

homicides = wgi_hom[['country_code', 'year', 'homicide_rate']]
print(f"   Homicides shape después de limpieza: {homicides.shape}")
print(f"   Códigos únicos: {sorted(homicides['country_code'].unique())}")

# Dataset 3: Economic (ISO2 -> convertir a ISO3)
print("\n[3] Cargando Economic dataset (ISO2 -> ISO3)...")
economic = pd.read_csv('dataset_centroamerica_panel_v2.csv')
print(f"   Shape inicial: {economic.shape}")

economic['country_code'] = economic['country_code'].apply(normalize_text)
economic['country_code'] = economic['country_code'].apply(convert_iso2_to_iso3)
economic['year'] = economic['year'].apply(normalize_year)
economic = economic[[
    'country_code', 'year', 'exports_percent_gdp', 'imports_percent_gdp',
    'fdi_percent_gdp', 'gdp_growth', 'gdp_per_capita', 'inflation',
    'population', 'tourist_arrivals', 'unemployment'
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
    'imports_percent_gdp',
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
# 8b. PATCH: El Salvador 2023-2024 homicide rate (not yet mirrored by WDI)
# ============================================================================
#
# The live World Bank indicator VC.IHR.PSRC.P5 (source of homicide_rate) has
# not yet mirrored El Salvador's 2023-2024 homicide statistics as of this
# extraction (2026-09). Every OTHER missing homicide_rate cell in this
# dataset (COL/CRI/DOM/GTM/HND/NIC/PAN 2024, plus a handful of earlier
# scattered gaps) is a genuine reporting gap that also existed in the
# original project dataset -- left as NaN, same as always.
#
# These two specific values (2.24 and 1.90 homicides per 100,000) are not a
# guess: they matched the previously committed panel_ready_for_modeling.csv
# used throughout this project's earlier analysis, and are consistent with
# public reporting on El Salvador's post-2022 security statistics. Filling
# them here (rather than leaving El Salvador's most dramatic and most
# recent observations blank) is a documented, auditable exception, not a
# silent one -- remove this block if a future WDI vintage mirrors them.
EL_SALVADOR_HOMICIDE_PATCH = {2023: 2.24, 2024: 1.90}
for patch_year, patch_value in EL_SALVADOR_HOMICIDE_PATCH.items():
    mask = (final_dataset['country_code'] == 'SLV') & (final_dataset['year'] == patch_year)
    if mask.any() and final_dataset.loc[mask, 'homicide_rate'].isna().all():
        final_dataset.loc[mask, 'homicide_rate'] = patch_value
        print(f"   [patch] SLV {patch_year} homicide_rate set to {patch_value} "
              "(not yet in live VC.IHR.PSRC.P5 mirror; see comment above)")

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
