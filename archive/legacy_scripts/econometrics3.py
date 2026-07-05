"""
Pipeline de Econometría Secuencial (2SLS/Recursive) - El Salvador
Versión Optimizada con: Detrending, Bootstrap, Diagnósticos y Random Forest/SHAP
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as smd
import statsmodels.stats.stattools as sst
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# SHAP es opcional — se importa con fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP no instalado. Se usará Feature Importance estándar. `pip install shap` para activarlo.")

np.random.seed(42)

# ─────────────────────────────────────────────
# SECCIÓN 0: CONFIGURACIÓN
# ─────────────────────────────────────────────
N_BOOTSTRAP = 1000        # Iteraciones bootstrap (aumentar a 2000+ para publicación)
ALPHA = 0.05              # Nivel de significancia
COUNTRY = "SLV"

print("=" * 65)
print("  PIPELINE ECONOMÉTRICO — EL SALVADOR: VIOLENCIA Y CRECIMIENTO")
print("=" * 65)

# ─────────────────────────────────────────────
# SECCIÓN 1: CARGA Y LIMPIEZA
# ─────────────────────────────────────────────
df = pd.read_csv("panel_ready_for_modeling.csv")
df = df.sort_values(["country_code", "year"])
df = df[df["country_code"] == COUNTRY].copy()
df = df.reset_index(drop=True)

print(f"\n✓ Datos cargados: {len(df)} observaciones para El Salvador ({df['year'].min()}–{df['year'].max()})")

# ─────────────────────────────────────────────
# SECCIÓN 2: PCA DE INSTITUCIONES
# ─────────────────────────────────────────────
inst_vars = ["rule_of_law", "control_corruption", "political_stability"]
X_inst = df[inst_vars].dropna()

scaler_pca = StandardScaler()
X_std = scaler_pca.fit_transform(X_inst)

pca = PCA(n_components=1)
pca_scores = pca.fit_transform(X_std)
df.loc[X_inst.index, "institution_index"] = pca_scores

# Asegurar dirección positiva (mayor score = mejores instituciones)
if pca.components_[0][0] < 0:
    df["institution_index"] *= -1

pca_var_explained = pca.explained_variance_ratio_[0]
print(f"\n✓ PCA Institucional: varianza explicada = {pca_var_explained:.1%}")
print(f"  Cargas: {dict(zip(inst_vars, np.abs(pca.components_[0])))}")

# ─────────────────────────────────────────────
# SECCIÓN 3: LAG Y DETRENDING (CORRECCIÓN MULTICOLINEALIDAD)
# ─────────────────────────────────────────────
df["homicide_lag1"] = df["homicide_rate"].shift(1)

# SOLUCIÓN AL CONDITION NUMBER ELEVADO:
# En lugar de usar `year` directamente (colineal con las predicciones acumuladas),
# centramos el tiempo creando `year_c` (year demeaned). Esto elimina la multicolinealidad
# estructural sin perder la tendencia temporal en la interpretación.
year_mean = df["year"].mean()
df["year_c"] = df["year"] - year_mean   # Año centrado → year_c=0 representa el año promedio

# Adicionalmente, creamos la tendencia lineal detrended de las variables principales.
# Esto permite que cada ecuación capture residuos cíclicos limpios de la tendencia secular.
def detrend_series(series, years):
    """OLS residuals after regressing out a linear time trend."""
    valid = ~(series.isna() | years.isna())
    y = series[valid].values
    t = years[valid].values
    t_c = t - t.mean()
    beta = np.polyfit(t_c, y, 1)
    residuals = y - np.polyval(beta, t_c)
    out = pd.Series(np.nan, index=series.index)
    out[valid] = residuals
    return out

df["homicide_lag1_dt"] = detrend_series(df["homicide_lag1"], df["year"])
df["institution_index_dt"] = detrend_series(df["institution_index"], df["year"])
df["fdi_percent_gdp_dt"] = detrend_series(df["fdi_percent_gdp"], df["year"])

# Dataset final para modelos
df_model = df.dropna(subset=[
    "homicide_lag1", "homicide_lag1_dt",
    "institution_index", "institution_index_dt",
    "fdi_percent_gdp", "fdi_percent_gdp_dt",
    "gdp_growth", "year_c"
]).copy().reset_index(drop=True)

N = len(df_model)
print(f"\n✓ Dataset para modelado: N={N} observaciones")

# ─────────────────────────────────────────────
# SECCIÓN 4: MODELOS OLS ROBUSTOS (HC1) — CON YEAR CENTRADO
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("BLOQUE A — MODELOS ESTRUCTURALES OLS ROBUSTOS")
print("─" * 65)

# MODELO 1: Violencia → Instituciones
m1 = smf.ols("institution_index ~ homicide_lag1 + year_c", data=df_model).fit(cov_type="HC1")
df_model["institution_predicted"] = m1.predict(df_model)

# MODELO 2: Instituciones (predichas) → IED
m2 = smf.ols("fdi_percent_gdp ~ institution_predicted + year_c", data=df_model).fit(cov_type="HC1")
df_model["fdi_predicted"] = m2.predict(df_model)

# MODELO 3: IED (predicha) + Instituciones (predicha) → Crecimiento PIB
m3 = smf.ols("gdp_growth ~ fdi_predicted + institution_predicted + year_c", data=df_model).fit(cov_type="HC1")

print(m1.summary())
print(m2.summary())
print(m3.summary())

# ─────────────────────────────────────────────
# SECCIÓN 5: PRUEBAS DIAGNÓSTICAS AUTOMÁTICAS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("BLOQUE B — DIAGNÓSTICOS DE SUPUESTOS")
print("─" * 65)

def run_diagnostics(model, model_name, df_data):
    """
    Corre Durbin-Watson, Breusch-Godfrey, Breusch-Pagan y Jarque-Bera
    sobre los residuos del modelo dado.
    """
    resid = model.resid
    fitted = model.fittedvalues
    exog = model.model.exog

    print(f"\n── Diagnósticos: {model_name} ──")

    # 1. Durbin-Watson (autocorrelación de orden 1)
    dw = durbin_watson(resid)
    dw_interp = "Sin autocorrelación" if 1.5 < dw < 2.5 else "⚠ Posible autocorrelación"
    print(f"  Durbin-Watson:        {dw:.3f}  → {dw_interp}")

    # 2. Breusch-Godfrey (autocorrelación hasta orden 2)
    try:
        bg_lm, bg_pval, _, _ = acorr_breusch_godfrey(model, nlags=2)
        bg_interp = "Sin autocorrelación" if bg_pval > ALPHA else "⚠ Autocorrelación detectada"
        print(f"  Breusch-Godfrey(2):   LM={bg_lm:.3f}, p={bg_pval:.3f}  → {bg_interp}")
    except Exception as e:
        print(f"  Breusch-Godfrey:      No calculable ({e})")

    # 3. Breusch-Pagan (heterocedasticidad)
    try:
        bp_lm, bp_pval, _, _ = het_breuschpagan(resid, exog)
        bp_interp = "Homocedasticidad" if bp_pval > ALPHA else "⚠ Heterocedasticidad detectada (HC1 correcto)"
        print(f"  Breusch-Pagan:        LM={bp_lm:.3f}, p={bp_pval:.3f}  → {bp_interp}")
    except Exception as e:
        print(f"  Breusch-Pagan:        No calculable ({e})")

    # 4. Jarque-Bera (normalidad de residuos)
    try:
        jb_stat, jb_pval, _, _ = jarque_bera(resid)
        jb_interp = "Residuos normales" if jb_pval > ALPHA else "⚠ No normalidad (robusto con HC1)"
        print(f"  Jarque-Bera:          JB={jb_stat:.3f}, p={jb_pval:.3f}  → {jb_interp}")
    except Exception as e:
        print(f"  Jarque-Bera:          No calculable ({e})")

    # 5. Condition Number
    cn = np.linalg.cond(exog)
    cn_interp = "Aceptable" if cn < 1000 else ("Moderada" if cn < 10000 else "⚠ Alta multicolinealidad")
    print(f"  Condition Number:     {cn:.1f}  → {cn_interp}")

run_diagnostics(m1, "Modelo 1: Violencia → Instituciones", df_model)
run_diagnostics(m2, "Modelo 2: Instituciones → IED", df_model)
run_diagnostics(m3, "Modelo 3: IED + Instituciones → PIB", df_model)

# ─────────────────────────────────────────────
# SECCIÓN 6: BOOTSTRAP (Corrección para N pequeño)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print(f"BLOQUE C — BOOTSTRAP PARAMÉTRICO (B={N_BOOTSTRAP})")
print("  (Técnica recomendada para series cortas: N={N})")
print("─" * 65)

def bootstrap_ols(formula, data, B=1000):
    """
    Bootstrap por remuestreo con reemplazo sobre las observaciones.
    Devuelve distribución de coeficientes.
    """
    coef_names = smf.ols(formula, data=data).fit().params.index.tolist()
    boot_coefs = {c: [] for c in coef_names}

    for _ in range(B):
        sample = data.sample(n=len(data), replace=True)
        try:
            m_boot = smf.ols(formula, data=sample).fit()
            for c in coef_names:
                boot_coefs[c].append(m_boot.params[c])
        except Exception:
            continue

    results = {}
    for c in coef_names:
        arr = np.array(boot_coefs[c])
        results[c] = {
            "mean":  arr.mean(),
            "se":    arr.std(),
            "ci_lo": np.percentile(arr, 2.5),
            "ci_hi": np.percentile(arr, 97.5),
            "p_val": 2 * min((arr > 0).mean(), (arr < 0).mean())  # p empírico bilateral
        }
    return results

boot1 = bootstrap_ols("institution_index ~ homicide_lag1 + year_c", df_model, N_BOOTSTRAP)
boot2 = bootstrap_ols("fdi_percent_gdp ~ institution_predicted + year_c", df_model, N_BOOTSTRAP)
boot3 = bootstrap_ols("gdp_growth ~ fdi_predicted + institution_predicted + year_c", df_model, N_BOOTSTRAP)

def print_bootstrap(boot_res, model_name):
    print(f"\n  {model_name}")
    print(f"  {'Variable':<25} {'Coef':>8} {'SE':>8} {'CI 2.5%':>9} {'CI 97.5%':>9} {'p-val':>7}")
    print("  " + "-" * 68)
    for var, stats in boot_res.items():
        sig = "*" if stats["p_val"] < 0.1 else ""
        print(f"  {var:<25} {stats['mean']:>8.4f} {stats['se']:>8.4f} "
              f"{stats['ci_lo']:>9.4f} {stats['ci_hi']:>9.4f} {stats['p_val']:>7.3f}{sig}")

print_bootstrap(boot1, "Modelo 1 Bootstrap: Violencia → Instituciones")
print_bootstrap(boot2, "Modelo 2 Bootstrap: Instituciones → IED")
print_bootstrap(boot3, "Modelo 3 Bootstrap: IED + Instituciones → Crecimiento")
print("\n  (*) p < 0.10 por distribución empírica bootstrap")

# ─────────────────────────────────────────────
# SECCIÓN 7: RANDOM FOREST + SHAP / FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("BLOQUE D — RANDOM FOREST: IMPORTANCIA DE VARIABLES (ML vs OLS)")
print("─" * 65)

# Variables de entrada para el RF (todas las disponibles del sistema)
rf_features = ["homicide_lag1", "institution_index", "fdi_percent_gdp", "year_c"]
rf_target   = "gdp_growth"

df_rf = df_model[rf_features + [rf_target]].dropna().copy()
X_rf = df_rf[rf_features].values
y_rf = df_rf[rf_target].values

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=3,           # Regularización agresiva para N pequeño
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)
rf_model.fit(X_rf, y_rf)

# Feature Importance estándar (Gini)
gini_imp = pd.Series(rf_model.feature_importances_, index=rf_features).sort_values(ascending=False)
print("\n  Feature Importance (Gini/Impurity):")
for feat, imp in gini_imp.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# Permutation Importance (más confiable con N pequeño)
perm_result = permutation_importance(rf_model, X_rf, y_rf, n_repeats=100, random_state=42)
perm_imp = pd.Series(perm_result.importances_mean, index=rf_features).sort_values(ascending=False)
print("\n  Permutation Importance (más robusto para N pequeño):")
for feat, imp in perm_imp.items():
    bar = "█" * max(0, int(imp * 40))
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# SHAP Values (si disponible)
if SHAP_AVAILABLE:
    print("\n  SHAP Values (TreeExplainer):")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_rf)
    shap_mean = pd.Series(np.abs(shap_values).mean(axis=0), index=rf_features).sort_values(ascending=False)
    for feat, sv in shap_mean.items():
        bar = "█" * int(sv / shap_mean.max() * 30)
        print(f"  {feat:<25} {sv:.4f}  {bar}")
else:
    print("\n  (SHAP no disponible — instalar con `pip install shap`)")

# Comparación de Rankings
print("\n  ── Tabla Comparativa de Rankings ──")
rank_gini = gini_imp.rank(ascending=False).astype(int)
rank_perm = perm_imp.rank(ascending=False).astype(int)
comparison = pd.DataFrame({"Gini Rank": rank_gini, "Permut. Rank": rank_perm})
if SHAP_AVAILABLE:
    comparison["SHAP Rank"] = shap_mean.rank(ascending=False).astype(int)
print(comparison.sort_values("Gini Rank").to_string())

# ─────────────────────────────────────────────
# SECCIÓN 8: VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("BLOQUE E — GENERANDO VISUALIZACIONES")
print("─" * 65)

years = df_model["year"].values
scaler_norm = MinMaxScaler()

fig = plt.figure(figsize=(18, 14))
fig.suptitle("El Salvador: Mecanismo de Transmisión — Violencia → Instituciones → IED → Crecimiento\n(2SLS Recursivo | OLS Robusto HC1 + Bootstrap)", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Paleta de colores
C = {"hom": "#d62728", "inst": "#1f77b4", "fdi": "#2ca02c", "gdp": "#ff7f0e", "pred": "#9467bd"}

# ── Subplot 1: Tendencia de Homicidios ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(years, df_model["homicide_rate"], color=C["hom"], lw=2, marker="o", ms=4)
ax1.set_title("Tasa de Homicidios (original)", fontsize=10)
ax1.set_ylabel("Hom/100k hab")
ax1.axvline(2016, color="gray", ls="--", alpha=0.6, label="Pico 2016")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── Subplot 2: Modelo 1 — Ajuste ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(years, df_model["institution_index"], color=C["inst"], zorder=3, s=40, label="Observado")
ax2.plot(years, df_model["institution_predicted"], color=C["pred"], lw=2, ls="--", label="Predicho M1")
ax2.set_title("M1: Violencia → Instituciones", fontsize=10)
ax2.set_ylabel("Índice Institucional (PCA)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Subplot 3: Modelo 2 — Ajuste ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(years, df_model["fdi_percent_gdp"], color=C["fdi"], zorder=3, s=40, label="Observado")
ax3.plot(years, df_model["fdi_predicted"], color=C["pred"], lw=2, ls="--", label="Predicho M2")
ax3.set_title("M2: Instituciones → IED", fontsize=10)
ax3.set_ylabel("IED (% PIB)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Subplot 4: Modelo 3 — Ajuste ──
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(years, df_model["gdp_growth"], color=C["gdp"], zorder=3, s=40, label="Observado")
ax4.plot(years, m3.predict(df_model), color=C["pred"], lw=2, ls="--", label="Predicho M3")
ax4.axhline(0, color="black", lw=0.8, alpha=0.5)
ax4.set_title("M3: IED + Inst → Crecimiento PIB", fontsize=10)
ax4.set_ylabel("Crecimiento PIB (%)")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# ── Subplot 5: Serie Normalizada (Mecanismo de Transmisión) ──
ax5 = fig.add_subplot(gs[1, 1:])
norm_data = scaler_norm.fit_transform(df_model[["homicide_rate", "institution_index", "fdi_percent_gdp", "gdp_growth"]].values)
labels_norm = ["Homicidios", "Instituciones (PCA)", "IED % PIB", "Crecimiento PIB"]
colors_norm = [C["hom"], C["inst"], C["fdi"], C["gdp"]]
for i, (label, color) in enumerate(zip(labels_norm, colors_norm)):
    ax5.plot(years, norm_data[:, i], label=label, color=color, lw=2, marker="o", ms=3)
ax5.set_title("Mecanismo de Transmisión — Series Normalizadas [0–1]", fontsize=10)
ax5.set_ylabel("Escala Min-Max")
ax5.legend(fontsize=8, loc="upper right")
ax5.grid(alpha=0.3)

# ── Subplot 6: Feature Importance RF (Gini vs Permutation) ──
ax6 = fig.add_subplot(gs[2, 0])
x_pos = np.arange(len(gini_imp))
ax6.bar(x_pos - 0.2, gini_imp.values, 0.35, label="Gini", color="#5c85d6")
ax6.bar(x_pos + 0.2, [perm_imp.get(f, 0) for f in gini_imp.index], 0.35, label="Permutación", color="#f4a261")
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f.replace("_", "\n") for f in gini_imp.index], fontsize=7)
ax6.set_title("Random Forest: Importancia\nde Variables (vs OLS Causal)", fontsize=10)
ax6.set_ylabel("Importancia")
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3, axis="y")

# ── Subplot 7: Bootstrap CI — coef homicide_lag1 en M1 ──
ax7 = fig.add_subplot(gs[2, 1])
boot_dist = []
for _ in range(N_BOOTSTRAP):
    sample = df_model.sample(n=N, replace=True)
    try:
        b = smf.ols("institution_index ~ homicide_lag1 + year_c", data=sample).fit()
        boot_dist.append(b.params["homicide_lag1"])
    except Exception:
        pass
ax7.hist(boot_dist, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
ax7.axvline(np.percentile(boot_dist, 2.5), color="red", ls="--", label="IC 95%")
ax7.axvline(np.percentile(boot_dist, 97.5), color="red", ls="--")
ax7.axvline(np.mean(boot_dist), color="navy", lw=2, label=f"Media: {np.mean(boot_dist):.3f}")
ax7.set_title("Bootstrap: β₁ Homicidios→Instituciones\n(Distribución Empírica)", fontsize=10)
ax7.set_xlabel("β₁")
ax7.legend(fontsize=8)
ax7.grid(alpha=0.3)

# ── Subplot 8: Residuos M3 ──
ax8 = fig.add_subplot(gs[2, 2])
resid_m3 = m3.resid
ax8.scatter(m3.fittedvalues, resid_m3, color="#d62728", s=40, alpha=0.7)
ax8.axhline(0, color="black", lw=1)
ax8.set_title("Residuos vs. Valores Ajustados\nModelo 3 (Crecimiento PIB)", fontsize=10)
ax8.set_xlabel("Valores Ajustados")
ax8.set_ylabel("Residuos")
ax8.grid(alpha=0.3)

plt.savefig("econometrics_sv_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Figura guardada como 'econometrics_sv_results.png'")

# ─────────────────────────────────────────────
# SECCIÓN 9: TABLA RESUMEN EJECUTIVO
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMEN EJECUTIVO — MECANISMO DE TRANSMISIÓN")
print("=" * 65)

models = [("M1: Violencia→Instituciones", m1, "homicide_lag1"),
          ("M2: Instituciones→IED",        m2, "institution_predicted"),
          ("M3: IED+Inst→PIB",             m3, "fdi_predicted")]

print(f"\n{'Modelo':<30} {'Variable Clave':<25} {'Coef':>8} {'p-val':>7} {'R²':>6} {'IC 95% bootstrap':>20}")
print("─" * 100)
for name, model, key_var in models:
    coef   = model.params[key_var]
    pval   = model.pvalues[key_var]
    r2     = model.rsquared_adj
    # Buscar en bootstrap
    boot_map = {"homicide_lag1": boot1, "institution_predicted": boot2, "fdi_predicted": boot3}
    if key_var in boot_map[key_var]:
        ci_lo = boot_map[key_var][key_var]["ci_lo"]
        ci_hi = boot_map[key_var][key_var]["ci_hi"]
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    else:
        ci_str = "—"
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"{name:<30} {key_var:<25} {coef:>8.4f} {pval:>7.3f}{sig} {r2:>6.3f} {ci_str:>20}")

print("\n  Significancia: *** p<0.01  ** p<0.05  * p<0.10")
print(f"  N={N} | Bootstrap B={N_BOOTSTRAP} | HC1 Robust SE | year centrado en {year_mean:.0f}")
print("=" * 65)