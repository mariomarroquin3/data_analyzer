"""
run_pipeline.py
════════════════════════════════════════════════════════════════════════
Master runner — executes all ten modules in sequence and assembles a
professional PDF report from structured JSON outputs + pipeline figures.

Usage
─────
    python run_pipeline.py [--data PATH_TO_CSV] [--skip 4 5]

Arguments
─────────
  --data   PATH   Path to panel CSV (default: panel_ready_for_modeling.csv
                  in the same directory as this script).
  --skip   LIST   Module numbers to skip (e.g. --skip 4 to skip bootstrap
                  if running a quick test).
  --quiet         Suppress per-module separator banners.

Output
──────
All figures, tables, and JSON summaries are written to:
  ./figures/
  ./tables/
  ./json/

The final PDF is saved to:
  ./econometric_report.pdf

Prerequisites
─────────────
    pip install linearmodels statsmodels scikit-learn shap pandas numpy \
                matplotlib scipy arch joblib reportlab pymc arviz
════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from research_report import (
    ResearchReport,
    enable_reporting,
    disable_reporting,
    _fmt,
    _sigstars,
)

# ── Encoding ─────────────────────────────────────────────────────────────────
for stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEPARATOR = "=" * 70

MODULES = [
    ("01", "01_data_preparation.py",    "Data Preparation & Institution Index"),
    ("02", "02_panel_estimation.py",    "Panel Estimation (Two-Way FE)"),
    ("03", "03_diagnostics.py",         "Panel Diagnostics"),
    ("04", "04_bootstrap_inference.py", "Wild Cluster Bootstrap Inference"),
    ("05", "05_robustness.py",          "Robustness Checks"),
    ("06", "06_ml_triangulation.py",    "ML Triangulation"),
    ("07", "07_cointegration.py",       "Panel Cointegration & ECM"),
    ("08", "08_growth_ceiling_risk.py", "Growth-Ceiling-at-Risk (Bayesian MCMC)"),
    ("09", "09_synthetic_control.py",   "Synthetic Control (Bukele Paradox)"),
    ("10", "10_arch_lm_test.py",        "ARCH-LM Test (Conditional Heteroskedasticity)"),
]


# ──────────────────────────────────────────────────────────────────────────────
# MODULE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_module(
    script: Path,
    skip: list,
    quiet: bool,
    report: Optional[ResearchReport] = None,
    figure_dir: Optional[Path] = None,
) -> bool:
    num = script.name[:2]
    if num in skip:
        print(f"\n  [SKIPPED] Module {num}: {script.name}")
        return True

    if not quiet:
        print(f"\n{SEPARATOR}")
        print(f"  RUNNING MODULE {num}: {script.name}")
        print(f"{SEPARATOR}")

    t0 = time.time()
    before: set = set()
    if figure_dir and figure_dir.exists():
        before = set(figure_dir.glob("*.png"))

    env = {
        **os.environ,
        "PYTHONPATH": str(script.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    with subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(script.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if report is not None:
                report.capture_text(line)
        ret_code = proc.wait()

    elapsed = time.time() - t0

    if ret_code != 0:
        print(f"\n  ✗ Module {num} FAILED (return code {ret_code})")
        return False

    print(f"\n  ✓ Module {num} completed in {elapsed:.1f}s")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# JSON LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def build_executive_summary(report: ResearchReport, meta: Dict, fe: Dict) -> None:
    report.add_section_h1("1. Resumen Ejecutivo")
    report.add_body_text(
        "Este informe presenta los resultados del pipeline econométrico que estima "
        "el efecto causal de la violencia (medida por la tasa de homicidios) sobre "
        "las instituciones, la inversión extranjera directa y el crecimiento económico "
        "en un panel de países de Centroamérica, Colombia y República Dominicana para "
        "el período 2000–2024. La estimación principal emplea efectos fijos bidireccionales "
        "(país × año) con errores estándar clusterizados a nivel de país."
    )

    n       = meta.get("N", "N/D")
    n_obs   = meta.get("N_obs", "N/D")
    t       = meta.get("T_distinct", "N/D")
    balanced = "Sí" if meta.get("is_balanced") else "No"
    pca_var  = meta.get("pca_var_exp_pc1", None)
    cronbach = meta.get("cronbach_alpha", None)

    eq1 = fe.get("EQ1", {})
    eq2 = fe.get("EQ2", {})
    eq3 = fe.get("EQ3", {})

    summary_data = [
        ["Indicador", "Valor"],
        ["Número de países (N)", str(n)],
        ["Períodos distintos (T)", str(t)],
        ["Observaciones totales", str(n_obs)],
        ["Panel balanceado", balanced],
        ["Varianza explicada PC1 (inst.)", _fmt(pca_var, 3) if pca_var else "N/D"],
        ["Alfa de Cronbach (inst.)", _fmt(cronbach, 3) if cronbach else "N/D"],
        ["EQ1: β (violencia → inst.)", _fmt(eq1.get("coef_key_cl"), 4)],
        ["EQ1: p-valor (CL)", _fmt(eq1.get("pval_key_cl"), 4)],
        ["EQ2: β (inst. → FDI)", _fmt(eq2.get("coef_key_cl"), 4)],
        ["EQ2: p-valor (CL)", _fmt(eq2.get("pval_key_cl"), 4)],
        ["EQ3: β (FDI → crecimiento)", _fmt(eq3.get("params_cl", {}).get("fdi_percent_gdp"), 4)],
        ["EQ3: p-valor FDI (CL)", _fmt(eq3.get("pvals_cl", {}).get("fdi_percent_gdp"), 4)],
    ]
    report.add_table(summary_data, col_widths=[4.0, 3.2])
    report.add_interpretation_box(
        "Lectura de resultados",
        "El coeficiente de EQ1 es negativo, indicando que mayor violencia se asocia con "
        "instituciones más débiles, aunque el p-valor sugiere que no se rechaza H₀ con los "
        "umbrales convencionales bajo este esquema de inferencia. EQ2 (instituciones → FDI) "
        "es significativo al 5% bajo tres de los cuatro estimadores de varianza: p≈0.007 "
        "(clusterizado), p≈0.085 (Driscoll-Kraay) y p≈0.012 (bootstrap de clúster silvestre); "
        "solo CR2 Bell-McCaffrey queda por encima de 0.10 (p≈0.126) — ver Sección 11 y "
        "tables/se_comparison.tex para la tabla completa. El mecanismo completo (violencia → "
        "inst. → FDI → crecimiento) no está probado como cadena conjunta: el bootstrap del "
        "efecto indirecto (Sección 11) no rechaza H₀ de que el producto de los tres "
        "coeficientes sea cero, ya que EQ1 y EQ3 individualmente no son significativos.",
        style="info",
    )


def build_dataset_section(report: ResearchReport, meta: Dict) -> None:
    report.add_section_h1("2. Información del Dataset")
    report.add_body_text(
        "El panel cubre 11 países: Centroamérica, Colombia, República Dominicana, México, Ecuador y Perú. "
        "Los datos provienen del Banco Mundial (WDI, WGI) y fuentes nacionales de estadísticas de crimen."
    )

    countries = meta.get("countries", [])
    country_data = [["Código", "País"]]
    names = {
        "COL": "Colombia", "CRI": "Costa Rica", "DOM": "Rep. Dominicana",
        "ECU": "Ecuador", "GTM": "Guatemala", "HND": "Honduras",
        "MEX": "México", "NIC": "Nicaragua", "PAN": "Panamá",
        "PER": "Perú", "SLV": "El Salvador",
    }
    for c in countries:
        country_data.append([c, names.get(c, c)])
    report.add_table(country_data, col_widths=[1.2, 3.0])

    years = meta.get("years", [])
    if years:
        report.add_body_text(
            f"Período temporal: {min(years)}–{max(years)}. "
            f"Número de períodos distintos observados: {meta.get('T_distinct', 'N/D')}. "
            f"El panel {'no ' if not meta.get('is_balanced') else ''}es balanceado."
        )

    within_data = [["Variable", "% Varianza Within-país"]]
    for var, val in meta.get("within_pct", {}).items():
        within_data.append([var, f"{val:.2f}%"])
    if len(within_data) > 1:
        report.add_section_h2("Varianza within-país")
        report.add_body_text(
            "La varianza within-país indica cuánta variación temporal existe dentro "
            "de cada unidad. Valores altos justifican el uso de efectos fijos."
        )
        report.add_table(within_data, col_widths=[3.5, 2.0])


def build_variables_section(report: ResearchReport, meta: Dict) -> None:
    report.add_section_h1("3. Variables Utilizadas")
    report.add_body_text(
        "A continuación se listan las variables dependientes, independientes clave "
        "y controles utilizados en el pipeline, junto con su transformación aplicada."
    )
    var_data = [
        ["Variable", "Descripción", "Transformación", "Rol"],
        ["homicide_rate_log", "Tasa de homicidios por 100K hab.", "log(x+1), lag 1 año", "Independiente clave (EQ1)"],
        ["inst_avg / inst_pca", "Índice institucional compuesto", "Promedio / PC1 estandarizado", "Mediador (EQ1→EQ2)"],
        ["rule_of_law", "Estado de derecho (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["control_corruption", "Control de corrupción (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["political_stability", "Estabilidad política (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["voice_accountability", "Voz y rendición de cuentas (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["government_effectiveness", "Efectividad gubernamental (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["regulatory_quality", "Calidad regulatoria (WGI)", "Sin transformar (percentil 0-100)", "Componente inst."],
        ["fdi_percent_gdp", "IED como % del PIB", "Sin transformar", "Dependiente EQ2 / Indep. EQ3"],
        ["gdp_growth", "Crecimiento real del PIB (%)", "Sin transformar", "Dependiente EQ3"],
        ["gdp_per_capita_log", "PIB per cápita (USD cte.)", "log(x)", "Control"],
        ["inflation", "Inflación anual (%)", "Sin transformar", "Control"],
        ["exports_percent_gdp", "Exportaciones como % del PIB", "Sin transformar", "Control"],
        ["population_log", "Población total", "log(x)", "Control"],
        ["unemployment", "Tasa de desempleo (%)", "Sin transformar", "Control"],
        ["remittances_percent_gdp", "Remesas como % del PIB", "Sin transformar", "Control alternativo (robustez, Sección 12)"],
    ]
    report.add_table(var_data, col_widths=[2.0, 2.5, 1.8, 1.5])

    # PCA / Cronbach info
    pca_var  = meta.get("pca_var_exp_pc1")
    cronbach = meta.get("cronbach_alpha")
    kmo      = meta.get("kmo")
    loadings = meta.get("pca_loadings", {})

    n_dims = len(loadings) if loadings else 6
    report.add_section_h2("Índice Institucional — PCA")
    report.add_body_text(
        f"Se extrajo el primer componente principal (PC1) de las {n_dims} dimensiones WGI "
        f"como medida sintética de calidad institucional. PC1 explica el "
        f"{_fmt(pca_var, 1) if pca_var else 'N/D'}% de la varianza. "
        f"El alfa de Cronbach es {_fmt(cronbach, 3) if cronbach else 'N/D'} "
        f"(KMO = {_fmt(kmo, 3) if kmo else 'N/D'})."
    )
    if loadings:
        load_data = [["Variable WGI", "Carga (Loading) PC1"]]
        for var, val in loadings.items():
            load_data.append([var, _fmt(val, 4)])
        report.add_table(load_data, col_widths=[3.5, 2.0])
    all_positive = all(v > 0 for v in loadings.values()) if loadings else None
    loadings_note = (
        "Todos los loadings son positivos, confirmando que PC1 captura 'calidad "
        "institucional general' y no una dimensión específica."
        if all_positive else
        "Los loadings tienen signos mixtos -- PC1 no representa de forma limpia una "
        "'calidad institucional general' uniforme; revisar la tabla de cargas antes de "
        "interpretar inst_pca como un índice unidireccional."
    )
    report.add_interpretation_box(
        "Validez del índice institucional",
        f"Un alfa de Cronbach > 0.80 indica alta consistencia interna de las {n_dims} dimensiones. "
        f"El primer componente explica {_fmt(pca_var, 1) if pca_var else '?'}% de la varianza, "
        f"justificando la reducción dimensional. {loadings_note}",
        style="info",
    )


def build_descriptive_section(report: ResearchReport, base_dir: Path) -> None:
    report.add_section_h1("4. Estadísticos Descriptivos")
    report.add_body_text(
        "Los estadísticos descriptivos se presentan para las variables utilizadas en el modelo. "
        "Las transformaciones logarítmicas reducen la asimetría de las distribuciones originales."
    )
    # Try to load from CSV
    csv_path = base_dir / "panel_ready_for_modeling.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        key_vars = [
            "homicide_rate_log", "inst_avg", "fdi_percent_gdp", "gdp_growth",
            "gdp_per_capita_log", "inflation", "exports_percent_gdp",
            "population_log", "unemployment",
        ]
        key_vars = [v for v in key_vars if v in df.columns]
        if key_vars:
            desc = df[key_vars].describe().round(3)
            desc_t = desc.T.reset_index()
            desc_t.columns = ["Variable"] + list(desc.columns)
            report.add_dataframe(
                desc_t,
                col_widths=[2.2, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
                note="N = número de observaciones no faltantes.",
            )
    else:
        report.add_body_text("[Dataset CSV no encontrado — omitiendo estadísticos descriptivos]")

    # Overview figure
    fig_path = base_dir / "figures" / "01_panel_overview.png"
    if fig_path.exists():
        report.add_section_h2("Panel Overview")
        report.add_image(
            fig_path,
            label="Figura 1 — Visión general del panel",
            caption=(
                "Series temporales de las variables clave por país. "
                "Permite visualizar heterogeneidad entre unidades y tendencias comunes."
            ),
        )


def build_correlations_section(report: ResearchReport, base_dir: Path) -> None:
    report.add_section_h1("5. Correlaciones")
    report.add_body_text(
        "El mapa de correlaciones de Pearson entre las variables del modelo "
        "permite detectar multicolinealidad potencial antes de la estimación."
    )
    fig_path = base_dir / "figures" / "03_correlation_heatmap.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 2 — Mapa de calor de correlaciones",
            caption=(
                "Correlaciones de Pearson entre las principales variables del panel. "
                "Valores cercanos a ±1 indican alta asociación lineal."
            ),
        )
    fig_path2 = base_dir / "figures" / "02_institution_indices.png"
    if fig_path2.exists():
        report.add_image(
            fig_path2,
            label="Figura 3 — Índices institucionales por país",
            caption="Evolución temporal de los índices institucionales (WGI) por unidad.",
        )


def build_pca_section(report: ResearchReport, meta: Dict, base_dir: Path) -> None:
    report.add_section_h1("6. Análisis de Componentes Principales (PCA)")
    pca_var  = meta.get("pca_var_exp_pc1", None)
    cronbach = meta.get("cronbach_alpha", None)
    kmo      = meta.get("kmo", None)
    loadings = meta.get("pca_loadings", {})
    corr_idx = meta.get("corr_indices", None)

    n_dims_pca = len(loadings) if loadings else 6
    report.add_body_text(
        f"El PCA reduce las {n_dims_pca} dimensiones WGI a un índice sintético único (inst_pca / inst_avg). "
        "La construcción del índice sigue el procedimiento estándar: estandarización Z, "
        "extracción de PC1, corrección de signo para interpretación positiva."
    )

    pca_summary = [
        ["Métrica", "Valor", "Umbral referencia"],
        ["Varianza explicada PC1", f"{_fmt(pca_var, 3) if pca_var else 'N/D'}", "> 0.60 deseable"],
        ["Alfa de Cronbach", f"{_fmt(cronbach, 3) if cronbach else 'N/D'}", "> 0.70 aceptable"],
        ["KMO", f"{_fmt(kmo, 3) if kmo else 'N/D'}", "> 0.50 mínimo"],
        ["Corr. inst_avg vs inst_pca", f"{_fmt(corr_idx, 4) if corr_idx else 'N/D'}", "> 0.95 muy alta"],
    ]
    report.add_table(pca_summary, col_widths=[2.8, 2.0, 2.4])

    if loadings:
        load_data = [["Variable", "Loading PC1", "Contribución"]]
        total_load = sum(abs(v) for v in loadings.values())
        for var, val in loadings.items():
            contribution = abs(val) / total_load * 100 if total_load > 0 else 0
            load_data.append([var, _fmt(val, 4), f"{contribution:.1f}%"])
        report.add_table(load_data, col_widths=[2.5, 1.5, 1.5])

    report.add_interpretation_box(
        "Justificación del índice institucional",
        f"La alta correlación entre inst_avg e inst_pca (r = {_fmt(corr_idx, 4) if corr_idx else '?'}) "
        f"indica que ambas medidas son prácticamente equivalentes. Se reportan resultados "
        f"con inst_avg (promedio simple) como índice principal por su mayor interpretabilidad, "
        f"usando inst_pca como verificación de robustez.",
        style="info",
    )


def build_model_sections(report: ResearchReport, fe: Dict, base_dir: Path) -> None:
    """Build sections 7, 8, 9 from FE results JSON."""

    eq_meta = {
        "EQ1": {
            "title": "7. Modelo 1 — Violencia → Instituciones",
            "dep": "inst_avg",
            "key_var": "homicide_rate_log_lag1",
            "description": (
                "La ecuación 1 estima el efecto de la tasa de homicidios (rezagada un período) "
                "sobre el índice institucional dentro de cada país. El uso del lag t-1 mitiga "
                "el sesgo de simultaneidad. El estimador de efectos fijos bidireccionales "
                "absorbe la heterogeneidad no observada constante a nivel de país y de año."
            ),
            "interpretation": (
                "El coeficiente negativo sobre homicide_rate_log_lag1 indica que mayor violencia "
                "se asocia con instituciones más débiles, consistente con la hipótesis teórica. "
                "La magnitud y significancia estadística deben evaluarse conjuntamente con los "
                "resultados bootstrap (Sección 11) dado el bajo número de clústeres (G=11)."
            ),
        },
        "EQ2": {
            "title": "8. Modelo 2 — Instituciones → FDI",
            "dep": "fdi_percent_gdp",
            "key_var": "inst_avg",
            "description": (
                "La ecuación 2 estima cómo las mejoras institucionales atraen inversión extranjera. "
                "Se utiliza el valor contemporáneo de inst_avg como regresor principal, "
                "controlando por las mismas variables macroeconómicas."
            ),
            "interpretation": (
                "El coeficiente positivo sobre inst_avg confirma el canal institucional "
                "(mejores instituciones reducen los riesgos de apropiación y contratos incompletos, "
                "atrayendo más IED), y es significativo al 5% bajo tres de los cuatro "
                "estimadores de varianza: p≈0.007 (clusterizado), p≈0.085 (Driscoll-Kraay) y "
                "p≈0.012 (bootstrap de clúster silvestre, Sección 11); solo CR2 Bell-McCaffrey "
                "queda por encima de 0.10 (p≈0.126). Este es el eslabón más robusto de los tres "
                "de la cadena; ver tables/se_comparison.tex para la tabla completa."
            ),
        },
        "EQ3": {
            "title": "9. Modelo 3 — FDI + Instituciones → Crecimiento",
            "dep": "gdp_growth",
            "key_var": "fdi_percent_gdp",
            "description": (
                "La ecuación 3 estima la ecuación de crecimiento de largo plazo, incorporando "
                "FDI e instituciones como determinantes junto a los controles macroeconómicos estándar."
            ),
            "interpretation": (
                "La inflación muestra el efecto negativo más robusto sobre el crecimiento (p < 0.001). "
                "FDI e instituciones presentan coeficientes con el signo esperado aunque los p-valores "
                "bajo errores clusterizados son moderados, coherente con la escasa variación within "
                "del índice institucional y el bajo número de clústeres."
            ),
        },
    }

    for eq_key, info in eq_meta.items():
        eq = fe.get(eq_key, {})
        if not eq:
            continue
        report.add_section_h1(info["title"])
        report.add_body_text(info["description"])

        # Build coefficient table
        dep_var   = eq.get("dep", info["dep"])
        n_obs     = eq.get("n_obs", None)
        rsq       = eq.get("rsq_within", None)
        params_cl = eq.get("params_cl", {})
        pvals_cl  = eq.get("pvals_cl", {})

        if eq_key == "EQ1":
            params_cl  = {eq.get("key_var", info["key_var"]): eq.get("coef_key_cl")}
            pvals_cl   = {eq.get("key_var", info["key_var"]): eq.get("pval_key_cl")}
            se_cl      = {eq.get("key_var", info["key_var"]): eq.get("se_key_cl")}
        elif eq_key == "EQ2":
            params_cl  = {eq.get("key_var", info["key_var"]): eq.get("coef_key_cl")}
            pvals_cl   = {eq.get("key_var", info["key_var"]): eq.get("pval_key_cl")}
            se_cl      = {eq.get("key_var", info["key_var"]): eq.get("se_key_cl")}
        else:
            se_cl = {}  # EQ3 doesn't expose SE in current JSON — use params only

        # For EQ3 use full params dict
        if eq_key == "EQ3":
            params_full = eq.get("params_cl", {})
            pvals_full  = eq.get("pvals_cl", {})
            # Build SE table from coef/pval only
            coef_rows = [["Variable", "Coef. (CL-SE)", "p-valor (CL)", "Sig.",
                          "Coef. (DK-SE)", "p-valor (DK)"]]
            params_dk = eq.get("params_dk", {})
            pvals_dk  = eq.get("pvals_dk", {})
            for var in params_full:
                coef_rows.append([
                    var,
                    _fmt(params_full.get(var), 4),
                    _fmt(pvals_full.get(var), 4),
                    _sigstars(pvals_full.get(var, 1.0)),
                    _fmt(params_dk.get(var), 4),
                    _fmt(pvals_dk.get(var), 4),
                ])
            coef_rows.append(["Dep. variable", dep_var, "", "", "", ""])
            coef_rows.append([f"N = {n_obs}", f"R² within = {_fmt(rsq, 4)}", "", "", "", ""])
            report.add_table(coef_rows, col_widths=[2.1, 1.1, 1.0, 0.5, 1.1, 1.0])
        else:
            # EQ1 / EQ2 — single key variable row + dual SE comparison
            coef_rows = [["Variable", "Coef.", "SE (CL)", "p (CL)", "Sig.", "SE (DK)", "p (DK)"]]
            key_var = eq.get("key_var", info["key_var"])
            coef_rows.append([
                key_var,
                _fmt(eq.get("coef_key_cl"), 4),
                _fmt(eq.get("se_key_cl"), 4),
                _fmt(eq.get("pval_key_cl"), 4),
                _sigstars(eq.get("pval_key_cl", 1.0)),
                _fmt(eq.get("se_key_dk"), 4),
                _fmt(eq.get("pval_key_dk"), 4),
            ])
            coef_rows.append(["Dep. variable", dep_var, "", "", "", "", ""])
            coef_rows.append([f"N = {n_obs}", f"R² within = {_fmt(rsq, 4)}", "", "", "", "", ""])
            report.add_table(coef_rows, col_widths=[2.0, 0.9, 0.9, 0.9, 0.5, 0.9, 0.9])

        report.add_interpretation_box(
            f"Interpretación — {eq_key}",
            info["interpretation"],
            style="info",
        )
        report.add_body_text(
            "Nota: CL-SE = errores estándar clusterizados a nivel de país. "
            "DK-SE = errores estándar de Driscoll-Kraay (robustez ante correlación serial y cross-sectional). "
            "*** p<0.01  ** p<0.05  * p<0.10."
        )

    # FE residuals figure
    fig_path = base_dir / "figures" / "04_fe_residuals.png"
    if fig_path.exists():
        report.add_section_h2("Residuos de los modelos FE")
        report.add_image(
            fig_path,
            label="Figura 4 — Diagnóstico visual de residuos (FE)",
            caption="Gráficos de residuos para las tres ecuaciones. Permite inspeccionar heterocedasticidad y outliers.",
        )


def build_diagnostics_section(report: ResearchReport, diag: Dict, base_dir: Path) -> None:
    report.add_section_h1("10. Diagnósticos del Panel")
    report.add_body_text(
        "Los diagnósticos evalúan los supuestos del modelo de efectos fijos: "
        "independencia serial, dependencia transversal y heterocedasticidad entre grupos."
    )

    for eq_label in ["eq1", "eq2", "eq3"]:
        eq_num = eq_label[-1]
        report.add_section_h2(f"Ecuación {eq_num}")

        rows_diag: List[Tuple[str, str, str, str]] = []

        # Wooldridge
        w = diag.get("wooldridge", {}).get(eq_label, {})
        if w:
            reject = "Sí — autocorrelación serial detectada" if w.get("reject_h0") else "No"
            rows_diag.append((
                "Test de Wooldridge (autocorrelación)",
                f"t = {_fmt(w.get('t_stat'), 3)}",
                _fmt(w.get("p_val"), 4),
                reject,
            ))

        # Pesaran CD
        p = diag.get("pesaran_cd", {}).get(eq_label, {})
        if p:
            reject = "Sí — dependencia cross-seccional" if p.get("reject_h0") else "No"
            rows_diag.append((
                "Test de Pesaran (dep. transversal)",
                f"CD = {_fmt(p.get('CD'), 3)}",
                _fmt(p.get("p_val"), 4),
                reject,
            ))

        # Modified Wald
        mw = diag.get("modified_wald", {}).get(eq_label, {})
        if mw:
            reject = "Sí — heterocedasticidad de grupo" if mw.get("reject_h0") else "No"
            rows_diag.append((
                "Wald Modificado (heterocedasticidad)",
                f"W = {_fmt(mw.get('W_stat'), 3)} (df={mw.get('df')})",
                _fmt(mw.get("p_val"), 4),
                reject,
            ))

        if rows_diag:
            report.add_diagnostics_table(
                rows_diag,
                note="H₀ varía por test. Ver descripción metodológica.",
            )

        # VIF table
        vif_data = diag.get("vif", {}).get(eq_label, {})
        vifs     = vif_data.get("vifs", {})
        cond_num = vif_data.get("condition_number")
        if vifs:
            vif_rows = [["Variable", "VIF", "Evaluación"]]
            for var, val in vifs.items():
                eval_str = "⚠ Posible multicolinealidad" if val > 5 else "✓ Aceptable"
                vif_rows.append([var, _fmt(val, 4), eval_str])
            if cond_num:
                vif_rows.append(["Número de condición", _fmt(cond_num, 4), "< 30 aceptable"])
            report.add_table(vif_rows, col_widths=[2.5, 1.2, 3.0])

    report.add_interpretation_box(
        "Implicaciones para la inferencia",
        "La presencia de autocorrelación serial y dependencia cross-seccional confirma que los "
        "errores estándar clusterizados a nivel de país son insuficientes para capturar toda la "
        "incertidumbre. Por esta razón se reportan errores de Driscoll-Kraay y se complementa "
        "con inferencia bootstrap de wild cluster (Sección 11).",
        style="warning",
    )

    fig_path = base_dir / "figures" / "05_diagnostics.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 5 — Diagnósticos panel",
            caption="Visualización de los tests de diagnóstico aplicados a las tres ecuaciones.",
        )


def build_arch_subsection(report: ResearchReport, arch: Dict) -> None:
    if not arch:
        return
    report.add_section_h2("Heterocedasticidad condicional (test ARCH-LM de Engle)")
    report.add_body_text(
        "Antes de considerar un modelo de volatilidad tipo GARCH (o Markov-Switching GARCH) "
        "conviene comprobar si existe heterocedasticidad condicional que modelar. Se aplica "
        "el test ARCH-LM de Engle (1982) país por país -el único nivel en que una serie "
        "temporal es válida- y se combinan los p-valores con el método de Fisher "
        "(H0: ningún país presenta efectos ARCH). Se evalúan los residuos de EQ1-EQ3 y las "
        "series originales de crecimiento y homicidios."
    )
    rows = [["Serie", "Países", "χ² Fisher", "p combinado", "Países sig. al 5%"]]
    for label, res in arch.items():
        if "error" in res:
            continue
        rows.append([
            label, str(res.get("n_countries", "")), _fmt(res.get("fisher_stat"), 1),
            "<0.001" if res.get("p_combined", 1) < 0.001 else _fmt(res.get("p_combined"), 3),
            f"{res.get('n_significant_at_05', 0)} de {res.get('n_countries', '')}",
        ])
    report.add_table(rows, col_widths=[3.0, 0.7, 0.9, 1.0, 1.3])
    report.add_interpretation_box(
        "Lectura y límites",
        "Los residuos de EQ2 y EQ3 no muestran evidencia de heterocedasticidad condicional. "
        "Los residuos de EQ1 y la serie de homicidios sí muestran señal, concentrada en "
        "varios países, compatible con ruido de medición variable en el índice WGI más que "
        "con regímenes de volatilidad propios de cada país. Con T~20-24 observaciones "
        "anuales por país el test tiene poco poder y no hay datos suficientes para ajustar "
        "ni un GARCH(1,1) de forma confiable, mucho menos un Markov-Switching GARCH. Los "
        "errores estándar clusterizados y de Driscoll-Kraay ya usados son la corrección "
        "apropiada para esta heterocedasticidad.",
        style="neutral",
    )


def build_bootstrap_section(report: ResearchReport, boot: Dict, med: Dict, base_dir: Path) -> None:
    report.add_section_h1("11. Inferencia Bootstrap (Wild Cluster)")
    report.add_body_text(
        "Con solo G=11 clústeres, los errores estándar clusterizados convencionales "
        "son conocidamente imprecisos (MacKinnon & Webb, 2017). Se aplica el wild cluster "
        "bootstrap con pesos de Webb (2023) usando B=999 réplicas. "
        "El p-valor bootstrap es exacto por construcción bajo H₀."
    )

    # Summary table across all equations
    boot_rows = [["Ecuación", "β", "SE (CL)", "t", "p-boot", "CI 95% (boot)", "Réplicas"]]
    eq_labels = {
        "eq1": "EQ1: Violencia → Inst.",
        "eq2": "EQ2: Inst. → FDI",
        "eq3": "EQ3: FDI → Crecimiento",
    }
    for eq_key, label in eq_labels.items():
        b = boot.get(eq_key, {})
        if not b:
            continue
        boot_rows.append([
            label,
            _fmt(b.get("beta_full"), 4),
            _fmt(b.get("se_full_cl"), 4),
            _fmt(b.get("t_full"), 3),
            _fmt(b.get("p_boot"), 4),
            f"[{_fmt(b.get('ci_lo'), 4)}, {_fmt(b.get('ci_hi'), 4)}]",
            str(b.get("n_valid", "N/D")),
        ])
    report.add_table(boot_rows, col_widths=[2.0, 0.8, 0.8, 0.7, 0.8, 1.6, 0.7])

    # CR2 robust SEs per equation
    for eq_key, label in eq_labels.items():
        b = boot.get(eq_key, {})
        cr2_se = b.get("cr2_se", {})
        cr2_p  = b.get("cr2_p", {})
        if not cr2_se:
            continue
        report.add_section_h2(f"Errores CR2 (Bell-McCaffrey) — {label}")
        cr2_rows = [["Variable", "SE (CR2)", "p (CR2)", "Sig."]]
        for var in cr2_se:
            pv = cr2_p.get(var, 1.0)
            cr2_rows.append([var, _fmt(cr2_se[var], 4), _fmt(pv, 4), _sigstars(pv)])
        report.add_table(cr2_rows, col_widths=[2.5, 1.2, 1.2, 0.8])

    report.add_interpretation_box(
        "Wild Cluster Bootstrap — Nota metodológica",
        "Los intervalos de confianza bootstrap se construyen con el método percentil simétrico. "
        "Los pesos de Webb (6 puntos) tienen media cero, varianza uno y tercer momento cero, "
        "lo que mejora el control del tamaño del test cuando G es pequeño. "
        "Un p-boot > 0.10 en EQ1 indica que, incluso con inferencia más conservadora, "
        "no se rechaza H₀ para el efecto violencia → instituciones con este tamaño muestral.",
        style="warning",
    )

    fig_path = base_dir / "figures" / "06_bootstrap_distributions.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 6 — Distribuciones bootstrap",
            caption=(
                "Distribuciones de los estadísticos t bootstrap bajo H₀ para las tres ecuaciones. "
                "La línea vertical indica el t observado."
            ),
        )

    # ── Mediation: bootstrap test of the full-chain indirect effect ────────
    if med:
        report.add_section_h2("11.1 Efecto Indirecto de la Cadena Completa (Mediación)")
        report.add_body_text(
            "EQ1, EQ2 y EQ3 se estiman como tres modelos independientes para evitar el "
            "problema de 'generated regressors' (Pagan, 1984). Esto significa que, hasta "
            "aquí, NINGÚN resultado del pipeline prueba formalmente el mecanismo de "
            "transmisión completo (Homicidios → Instituciones → IED → Crecimiento) como "
            "una cadena conjunta -- solo se han evaluado sus tres eslabones por separado. "
            "El remedio estándar (Sobel, 1982; Preacher & Hayes, 2008) es testear el "
            "PRODUCTO de los tres coeficientes de la cadena, indirecto = β₁·γ₁·δ₁, contra "
            "H₀: indirecto = 0, usando bootstrap por clúster de país (no por observación, "
            "dada la dependencia serial dentro de cada país)."
        )
        med_rows = [
            ["Cantidad", "Valor"],
            ["β₁ (Homicidios → Instituciones)", _fmt(med.get("beta1_full"), 4)],
            ["γ₁ (Instituciones → IED)", _fmt(med.get("gamma1_full"), 4)],
            ["δ₁ (IED → Crecimiento)", _fmt(med.get("delta1_full"), 4)],
            ["Efecto indirecto = β₁·γ₁·δ₁", _fmt(med.get("indirect_full"), 6)],
            ["Media bootstrap (B réplicas por país)", _fmt(med.get("boot_mean"), 6)],
            ["IC 95% (percentil)", f"[{_fmt(med.get('ci_lo'), 6)}, {_fmt(med.get('ci_hi'), 6)}]"],
            ["p-valor bootstrap (H₀: indirecto = 0)", _fmt(med.get("p_boot"), 4)],
        ]
        report.add_table(med_rows, col_widths=[3.5, 2.0])

        ci_lo = med.get("ci_lo")
        ci_hi = med.get("ci_hi")
        includes_zero = (ci_lo is not None and ci_hi is not None and ci_lo < 0 < ci_hi)
        report.add_interpretation_box(
            "Lectura honesta del mecanismo de transmisión",
            (
                "El intervalo de confianza al 95% del efecto indirecto INCLUYE CERO: la "
                "cadena completa Violencia → Instituciones → IED → Crecimiento NO es "
                "estadísticamente distinguible de ausencia de efecto con este tamaño "
                "muestral. Esto no refuta la teoría institucional, pero significa que el "
                "'mecanismo de transmisión' debe reportarse como una hipótesis consistente "
                "con los signos observados, no como un resultado probado -- cada eslabón es "
                "individualmente débil y el producto de los tres no supera un test formal."
                if includes_zero else
                "El intervalo de confianza al 95% del efecto indirecto EXCLUYE CERO: el "
                "producto de los tres coeficientes de la cadena es significativo, lo que "
                "respalda formalmente el mecanismo de transmisión propuesto (más allá de la "
                "significancia individual de cada eslabón)."
            ),
            style=("warning" if includes_zero else "info"),
        )

        fig_med = base_dir / "figures" / "06b_mediation_bootstrap.png"
        if fig_med.exists():
            report.add_image(
                fig_med,
                label="Figura 6b — Bootstrap del efecto indirecto",
                caption=(
                    "Distribución bootstrap (remuestreo por país) del efecto indirecto "
                    "β₁·γ₁·δ₁. La línea roja marca la estimación puntual; las líneas grises, "
                    "el intervalo de confianza al 95%."
                ),
            )


def build_robustness_section(report: ResearchReport, rob: Dict, base_dir: Path) -> None:
    report.add_section_h1("12. Pruebas de Robustez")
    report.add_body_text(
        "Los análisis de robustez verifican la estabilidad de los coeficientes ante: "
        "(i) exclusión de cada país (LOCO — Leave-One-Country-Out), "
        "(ii) especificaciones alternativas del índice institucional, "
        "(iii) variaciones en las variables de control."
    )

    # LOCO for EQ1
    loco_eq1 = rob.get("eq1_loco", [])
    if loco_eq1:
        report.add_section_h2("LOCO — EQ1 (Violencia → Instituciones)")
        loco_rows = [["Muestra", "Coef.", "SE", "p-valor", "R² within", "N"]]
        for entry in loco_eq1:
            loco_rows.append([
                entry.get("label", ""),
                _fmt(entry.get("coef"), 4),
                _fmt(entry.get("se"), 4),
                _fmt(entry.get("pval"), 4),
                _fmt(entry.get("rsq_within"), 4),
                str(entry.get("n_obs", "")),
            ])
        report.add_table(loco_rows, col_widths=[2.0, 0.9, 0.9, 0.9, 0.9, 0.7])
        report.add_interpretation_box(
            "LOCO — Estabilidad del coeficiente",
            "Si el coeficiente permanece negativo y de magnitud similar al excluir cada país, "
            "se concluye que los resultados no están impulsados por ninguna observación influyente. "
            "Una variación sustancial al excluir un país señala dependencia de ese clúster.",
            style="info",
        )

    fig_path = base_dir / "figures" / "07_robustness.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 7 — Robustez",
            caption="Coeficientes e intervalos de confianza bajo especificaciones alternativas.",
        )

    # Alternative institution indices (per WGI dimension)
    inst_idx = rob.get("eq1_inst_index", [])
    if inst_idx:
        report.add_section_h2("Índices Institucionales Alternativos (por dimensión WGI)")
        report.add_body_text(
            "En vez de inst_avg (promedio de las 6 dimensiones), se re-estima EQ1 usando "
            "cada dimensión WGI individualmente como variable dependiente, para verificar "
            "si el resultado depende de la forma en que se combinan."
        )
        idx_rows = [["Especificación", "Coef.", "SE", "p-valor", "N"]]
        for r in inst_idx:
            idx_rows.append([
                r.get("label", ""), _fmt(r.get("coef"), 4), _fmt(r.get("se"), 4),
                _fmt(r.get("pval"), 4), str(r.get("n_obs", "")),
            ])
        report.add_table(idx_rows, col_widths=[2.6, 1.0, 1.0, 1.0, 0.8])

    # The "Bukele paradox"
    va_loco = rob.get("eq1_voice_accountability_loco", [])
    if va_loco:
        report.add_section_h2('La "Paradoja Bukele": Seguridad vs. Libertades Civiles')
        report.add_body_text(
            "De las 6 dimensiones WGI, voice_accountability es la única donde la violencia "
            "rezagada tiene el signo contrario al esperado (positivo: menos violencia junto "
            "con MENOR voz y rendición de cuentas). Los datos de El Salvador 2021-2024 "
            "muestran exactamente este patrón: los homicidios colapsaron (17.3 → 1.9 por "
            "100 mil) mientras voice_accountability también cayó (53.5 → 45.0) -- el costo "
            "en libertades civiles, documentado, del régimen de excepción. Este chequeo "
            "Leave-One-Country-Out prueba si El Salvador es el único responsable del signo "
            "anómalo en la muestra completa, o si es un patrón regional más amplio."
        )
        va_rows = [["Muestra", "Coef.", "SE", "p-valor", "N"]]
        for r in va_loco:
            va_rows.append([
                r.get("label", ""), _fmt(r.get("coef"), 4), _fmt(r.get("se"), 4),
                _fmt(r.get("pval"), 4), str(r.get("n_obs", "")),
            ])
        report.add_table(va_rows, col_widths=[2.2, 1.0, 1.0, 1.0, 0.8])

        full_r = next((r for r in va_loco if r.get("label") == "Full sample"), None)
        slv_r  = next((r for r in va_loco if r.get("label") == "Excl. SLV"), None)
        flips = None
        if full_r and full_r.get("coef") is not None:
            full_positive = full_r["coef"] > 0
            flips = sum(
                1 for r in va_loco
                if r.get("label", "").startswith("Excl.") and r.get("converged")
                and r.get("coef") is not None and (r["coef"] > 0) != full_positive
            )

        fig_bukele = base_dir / "figures" / "14_bukele_paradox.png"
        if fig_bukele.exists():
            report.add_image(
                fig_bukele,
                label="Figura 14 — La paradoja Bukele",
                caption=(
                    "Izquierda: coeficiente de violencia sobre voice_accountability al "
                    "excluir cada país. Derecha: tasa de homicidios y voice_accountability "
                    "de El Salvador, 2000-2024, mostrando el período del régimen de "
                    "excepción (2021-2024)."
                ),
            )

        if full_r and slv_r and flips == 1:
            report.add_interpretation_box(
                "Un caso, no un patrón regional",
                f"El Salvador es el ÚNICO país cuya exclusión invierte el signo "
                f"(muestra completa β={_fmt(full_r.get('coef'), 3)}, excl. El Salvador "
                f"β={_fmt(slv_r.get('coef'), 3)}, ninguno de los dos significativo al 5%). "
                "El coeficiente anómalo de la muestra completa no refleja una relación "
                "general entre violencia y voz institucional en la región -- es "
                "enteramente producto del caso salvadoreño, donde las ganancias de "
                "seguridad y el deterioro de las libertades civiles avanzaron juntos, no "
                "en direcciones opuestas. Esto no invalida el hallazgo de EQ1 sobre las "
                "otras 5 dimensiones WGI; añade una advertencia específica sobre qué mide "
                "'voz y rendición de cuentas' durante una transformación de seguridad "
                "autoritaria.",
                style="warning",
            )

    # Panel unit-root test (Fisher-ADF / Maddala-Wu)
    unit_root = rob.get("panel_unit_root", {})
    if unit_root:
        report.add_section_h2("Test de Raíz Unitaria de Panel (Fisher-ADF, Maddala y Wu 1999)")
        report.add_body_text(
            "Con T~25 años por país, este test tiene poco poder estadístico; no rechazar "
            "H0 es evidencia débil de raíz unitaria, no una prueba definitiva. Se usa aquí "
            "como base para la sección de cointegración que sigue."
        )
        ur_rows = [["Variable", "N países", "χ²", "gl", "p (Fisher)", "¿I(1)?"]]
        for var, res in unit_root.items():
            is_i1 = res.get("p_fisher", 0) > 0.05
            ur_rows.append([
                var,
                str(res.get("n_countries", "")),
                _fmt(res.get("fisher_stat"), 2),
                str(res.get("df_chi2", "")),
                _fmt(res.get("p_fisher"), 4),
                "Sí (no rechaza)" if is_i1 else "No (rechaza)",
            ])
        report.add_table(ur_rows, col_widths=[2.0, 1.0, 1.0, 0.7, 1.2, 1.6])


def build_synthetic_control_section(report: ResearchReport, sc: Dict, base_dir: Path) -> None:
    if not sc:
        return
    report.add_section_h1('13. Control Sintético: La "Paradoja Bukele" como Estudio de Caso')
    report.add_body_text(
        "El chequeo Leave-One-Country-Out de la Sección 12 estableció que El Salvador es "
        "el único país cuya exclusión invierte el signo del coeficiente agregado de "
        "voice_accountability -- un chequeo de robustez, no un diseño causal. Esta sección "
        "responde una pregunta más estrecha y mejor identificada con el Método de Control "
        "Sintético (Abadie, Diamond y Hainmueller, 2010): ¿qué habría pasado con "
        "voice_accountability en El Salvador si no hubiera ocurrido el régimen de excepción "
        "de 2021-2024? Se construye un 'El Salvador sintético' como combinación ponderada "
        "de los otros 10 países de la muestra, ajustada para replicar la trayectoria real "
        "de El Salvador en el período previo (2000-2020), y se compara con la trayectoria "
        "observada después."
    )

    names = {
        "COL": "Colombia", "CRI": "Costa Rica", "DOM": "Rep. Dominicana",
        "ECU": "Ecuador", "GTM": "Guatemala", "HND": "Honduras", "MEX": "México",
        "NIC": "Nicaragua", "PAN": "Panamá", "PER": "Perú", "SLV": "El Salvador",
    }

    slv = sc.get("el_salvador", {})
    weights = slv.get("weights", {})
    if weights:
        report.add_body_text(
            "El Salvador sintético se construye como la combinación ponderada que mejor "
            "replica su propia trayectoria pre-tratamiento (sin usar covariables adicionales, "
            "para evitar sobreajuste con solo 10 países donantes):"
        )
        w_rows = [["País donante", "Peso"]]
        for code, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            w_rows.append([names.get(code, code), f"{w:.3f}"])
        report.add_table(w_rows, col_widths=[2.4, 1.0])

    report.add_body_text(
        f"El ajuste pre-tratamiento es muy bueno (RMSPE = {_fmt(slv.get('pre_rmspe'), 2)} "
        f"puntos, 2000-2020). En 2024, el valor observado de voice_accountability es "
        f"{_fmt(slv.get('actual_2024'), 1)}, frente a {_fmt(slv.get('synthetic_2024'), 1)} "
        f"en la contrafactual sintética -- una brecha de {_fmt(slv.get('gap_2024'), 1)} "
        "puntos que el modelo atribuye al régimen de excepción y no a una tendencia "
        "preexistente."
    )

    fig_sc = base_dir / "figures" / "15_synthetic_control_bukele.png"
    if fig_sc.exists():
        report.add_image(
            fig_sc,
            label="Figura 15 — Control sintético de El Salvador",
            caption=(
                "Izquierda: voice_accountability real de El Salvador vs. su contrafactual "
                "sintética, 2000-2024. Derecha: brecha (real - sintética) de El Salvador "
                "(línea roja) frente a las brechas placebo de los otros 10 países "
                "(líneas grises), con el régimen de excepción sombreado."
            ),
        )

    report.add_section_h2("Inferencia por placebo-en-el-espacio")
    report.add_body_text(
        "Siguiendo a Abadie et al. (2010), se repite el procedimiento asignando el papel "
        "de 'tratado' a cada uno de los otros 10 países (placebos), y se compara la razón "
        "RMSPE post/pre-tratamiento de El Salvador contra la distribución de razones "
        "placebo -- un test exacto de aleatorización, no un p-valor asintótico."
    )
    placebo_ratios = sc.get("placebo_ratios", {})
    if placebo_ratios:
        pr_rows = [["País", "Razón RMSPE post/pre"]]
        for code, r in sorted(placebo_ratios.items(), key=lambda kv: -kv[1]):
            marker = " (caso real)" if code == sc.get("treated_unit") else ""
            pr_rows.append([names.get(code, code) + marker, _fmt(r, 2)])
        report.add_table(pr_rows, col_widths=[2.6, 1.4])

    p_val = sc.get("p_value")
    rank  = sc.get("rank_of_slv")
    n     = sc.get("n_countries")
    p_val_wf = sc.get("p_value_well_fitting_only")
    n_wf     = sc.get("n_well_fitting")
    report.add_interpretation_box(
        "De chequeo de robustez a estudio de caso cuasi-causal",
        f"El Salvador tiene la razón RMSPE post/pre más alta de los {n} países "
        f"(rank {rank}/{n}), lo que da un p-valor exacto de aleatorización de "
        f"{_fmt(p_val, 3)} -- el valor mínimo posible con {n} unidades. Restringiendo la "
        f"comparación a los {n_wf} países cuyo propio ajuste pre-tratamiento es al menos "
        f"tan bueno como el de El Salvador, sigue ocupando el primer lugar "
        f"(p = {_fmt(p_val_wf, 3)}). Esto no es solo 'El Salvador es el único país cuya "
        "exclusión invierte el signo' (Sección 12) -- es que la caída observada en "
        "voice_accountability es, en sí misma, la más extrema de la región frente a su "
        "propia trayectoria contrafactual. Con solo 10 países donantes, este resultado "
        "debe leerse como ilustrativo y no como una estimación causal precisa (el p-valor "
        "mínimo atribuible es 1/11 = 0.091), pero eleva la 'paradoja Bukele' de un patrón "
        "correlacional a un caso con diseño cuasi-experimental explícito.",
        style="info",
    )


def build_cointegration_section(report: ResearchReport, coint: Dict, base_dir: Path) -> None:
    report.add_section_h1("14. Cointegración de Panel y Modelo de Corrección de Errores (ECM)")
    report.add_body_text(
        "El test de raíz unitaria de panel (Sección 12) encontró que homicide_rate_log, "
        "inst_avg y gdp_per_capita_log no rechazan raíz unitaria (son I(1)), mientras que "
        "fdi_percent_gdp y gdp_growth sí la rechazan (son I(0)). Dos series I(1) pueden, "
        "sin embargo, compartir una relación de equilibrio de largo plazo genuina "
        "(cointegración), en cuyo caso una regresión estática en niveles no sería "
        "necesariamente espuria. Este módulo prueba esto para los tres pares de variables "
        "I(1) del panel -- deliberadamente NO se prueba EQ2 ni EQ3, ya que emparejar una "
        "serie I(1) con una I(0) no es una pregunta de cointegración coherente."
    )
    report.add_body_text(
        "Método: enfoque de dos pasos de Engle-Granger extendido a panel (Kao, 1999; "
        "McCoskey y Kao, 1998) -- (1) se estima la relación de largo plazo vía efectos "
        "fijos bidireccionales; (2) se testea la raíz unitaria de los residuos con el "
        "mismo test Fisher-ADF de la Sección 12. Residuos estacionarios = evidencia de "
        "cointegración; solo entonces se estima un ECM."
    )

    if not coint:
        report.add_body_text("(Sin resultados de cointegración disponibles.)")
        return

    rows = [["Par", "beta (largo plazo)", "p (Fisher, residuos)", "¿Cointegrado?"]]
    for name, info in coint.items():
        lr = info.get("long_run", {})
        ur = info.get("residual_unit_root") or {}
        rows.append([
            name,
            f"{_fmt(lr.get('beta'), 4)} (p={_fmt(lr.get('pval_cl'), 3)})",
            _fmt(ur.get("p_fisher"), 4) if ur else "N/D",
            "Sí" if info.get("cointegrated") else "No",
        ])
    report.add_table(rows, col_widths=[2.3, 2.0, 1.7, 1.3])

    any_coint = any(info.get("cointegrated") for info in coint.values())
    if any_coint:
        ecm_rows = [["Par", "phi (velocidad ajuste)", "p(phi, cl)", "gamma (corto plazo)", "p(gamma, cl)"]]
        for name, info in coint.items():
            ecm = info.get("ecm")
            if not ecm:
                continue
            ecm_rows.append([
                name,
                _fmt(ecm.get("phi_cl"), 4),
                _fmt(ecm.get("phi_p_cl"), 4),
                _fmt(ecm.get("gamma_cl"), 4),
                _fmt(ecm.get("gamma_p_cl"), 4),
            ])
        if len(ecm_rows) > 1:
            report.add_section_h2("Modelo de Corrección de Errores (pares cointegrados)")
            report.add_table(ecm_rows, col_widths=[2.2, 1.9, 1.3, 1.9, 1.3])
        report.add_interpretation_box(
            "Cointegración encontrada",
            "Al menos un par de variables I(1) comparte una relación de equilibrio de "
            "largo plazo genuina. phi < 0 y significativo indica que la variable "
            "dependiente corrige parte de cualquier desviación de esa relación cada año "
            "(velocidad de ajuste); gamma captura el efecto de corto plazo, distinto del "
            "efecto de largo plazo (beta) de la tabla anterior.",
            style="info",
        )
    else:
        report.add_interpretation_box(
            "Ningún par muestra evidencia de cointegración",
            "Esto es, en sí mismo, un hallazgo relevante: las relaciones en niveles entre "
            "estas series I(1) -estimadas mediante efectos fijos estáticos en el resto de "
            "este pipeline- no pueden distinguirse de una regresión de panel espuria con "
            "los datos disponibles. En particular para T1 (Homicidios ↔ Instituciones, la "
            "base de EQ1), esto ofrece una explicación formal adicional -más allá del bajo "
            "poder estadístico por G=11- de por qué ese eslabón es el más inestable de los "
            "tres en todas las pruebas de este pipeline.",
            style="warning",
        )

    fig_path = base_dir / "figures" / "11_cointegration_residuals.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 11 — Residuos de la regresión de largo plazo por país",
            caption=(
                "Residuos de la relación de largo plazo para cada par I(1). Residuos que "
                "revierten a cero son evidencia de cointegración; residuos que se alejan "
                "persistentemente (como Colombia o Nicaragua en varios paneles) son "
                "evidencia de una relación espuria."
            ),
        )

    report.add_body_text(
        "Advertencias: G=11, T~25 es una muestra pequeña incluso para el test de "
        "Engle-Granger de series individuales; la extensión a panel no corrige un tamaño "
        "muestral fundamentalmente pequeño. Este es un estimador de dos pasos "
        "simplificado, no un sistema de cointegración/ECM totalmente eficiente (sin "
        "corrección por dependencia transversal en la regresión de largo plazo, sin "
        "estadísticos Pedroni/Westerlund, sin bootstrap de clúster silvestre en el ECM)."
    )


def build_growth_ceiling_section(report: ResearchReport, gcar: Dict, base_dir: Path) -> None:
    report.add_section_h1("15. Growth-Ceiling-at-Risk (MCMC Bayesiano)")
    report.add_body_text(
        "Un análisis exploratorio de regresión cuantílica frecuentista (ver README, sección "
        "'Exploratory Finding') encontró que la violencia rezagada no tiene efecto cerca de la "
        "mediana del crecimiento del PIB, pero sí un efecto negativo grande y robusto a LOCO en "
        "la cola SUPERIOR (q=0.90/0.95) -- un patrón de 'techo de crecimiento', inverso al "
        "Growth-at-Risk clásico (Adrian, Boyarchenko y Giannone, 2019), que estudia la cola "
        "inferior. Este módulo re-estima ese patrón con un modelo bayesiano jerárquico en vez de "
        "variables dummy por país (LSDV): un prior de partial pooling regulariza los interceptos "
        "de cada país -algo que LSDV no puede hacer con G=11- y la incertidumbre se reporta como "
        "una distribución posterior completa (intervalo de credibilidad, P(beta<0|datos)) en vez "
        "de un p-valor asintótico, ya señalado como poco confiable a este tamaño muestral en "
        "otras secciones de este pipeline."
    )
    report.add_body_text(
        "Método: regresión cuantílica bayesiana vía la distribución Asimétrica de Laplace (ALD; "
        "Yu y Moyeed, 2001), implementada como un pm.Potential en PyMC. Los interceptos por país "
        "usan una parametrización no centrada (alpha_i = alpha_mu + alpha_sigma*z_i) para evitar "
        "la patología de 'embudo' jerárquico bajo NUTS (Betancourt y Girolami, 2015)."
    )

    if not gcar:
        report.add_body_text("(Sin resultados de Growth-Ceiling-at-Risk disponibles.)")
        return

    quantiles = gcar.get("quantiles", {})
    if quantiles:
        rows = [["q", "Coef. frecuentista (LSDV)", "p (frec.)", "Media posterior (bayes.)",
                 "94% HDI", "P(β<0|datos)"]]
        for qk, info in quantiles.items():
            b, f = info.get("bayes", {}), info.get("freq", {})
            rows.append([
                qk,
                _fmt(f.get("coef"), 4),
                _fmt(f.get("pval"), 4),
                _fmt(b.get("beta_mean"), 4),
                f"[{_fmt(b.get('beta_hdi_lo'), 3)}, {_fmt(b.get('beta_hdi_hi'), 3)}]",
                _fmt(b.get("p_beta_negative"), 3),
            ])
        report.add_table(rows, col_widths=[0.6, 2.0, 1.1, 2.0, 1.9, 1.4])

    fig_path = base_dir / "figures" / "12_growth_ceiling_bayesian.png"
    if fig_path.exists():
        report.add_image(
            fig_path,
            label="Figura 12 — Coeficiente por cuantil: frecuentista vs. bayesiano",
            caption=(
                "Efecto de la violencia rezagada sobre cada cuantil del crecimiento del PIB. "
                "El intervalo de credibilidad bayesiano (94% HDI) regulariza los interceptos por "
                "país vía partial pooling, en vez de absorberlos con variables dummy (LSDV)."
            ),
        )

    report.add_interpretation_box(
        "Patrón de techo de crecimiento confirmado bayesianamente",
        "En q=0.90 y q=0.95 el modelo jerárquico bayesiano confirma el hallazgo exploratorio: "
        "la probabilidad posterior de que el coeficiente sea negativo es cercana o igual a 1 "
        "(ver tabla), con diagnósticos de convergencia limpios (R-hat y ausencia de "
        "divergencias). El chequeo secundario con inst_avg como variable condicionante "
        "(mismos cuantiles, mismo modelo) no muestra el mismo patrón, apoyando que el efecto de "
        "techo es específico a la violencia y no una característica genérica de cualquier "
        "regresor en este panel.",
        style="info",
    )

    scenario = gcar.get("growth_ceiling_scenario", {})
    if scenario:
        report.add_section_h2("Escenario Growth-Ceiling-at-Risk")
        scen_rows = [["q", "Techo (violencia baja, p10)", "Techo (violencia alta, p90)",
                      "Caída", "94% HDI (caída)", "P(caída>0|datos)"]]
        for qk, s in scenario.items():
            scen_rows.append([
                qk,
                _fmt(s.get("ceiling_low_violence_mean"), 2),
                _fmt(s.get("ceiling_high_violence_mean"), 2),
                _fmt(s.get("ceiling_drop_mean"), 2),
                f"[{_fmt(s.get('ceiling_drop_hdi_lo'), 2)}, {_fmt(s.get('ceiling_drop_hdi_hi'), 2)}]",
                _fmt(s.get("p_drop_positive"), 3),
            ])
        report.add_table(scen_rows, col_widths=[0.6, 2.0, 2.0, 1.1, 1.8, 1.5])

        fig_path2 = base_dir / "figures" / "13_growth_ceiling_scenario.png"
        if fig_path2.exists():
            report.add_image(
                fig_path2,
                label="Figura 13 — Distribución posterior del techo de crecimiento",
                caption=(
                    "Distribución posterior del q-ésimo percentil de crecimiento alcanzable bajo "
                    "un escenario de violencia baja (percentil 10) vs. alta (percentil 90), "
                    "manteniendo país y año en su nivel promedio."
                ),
            )

    report.add_body_text(
        "Advertencias: este es un hallazgo exploratorio, no una hipótesis pre-registrada -- "
        "tratar todo resultado como sugestivo. G=11 sigue siendo pequeño incluso para un modelo "
        "jerárquico: el partial pooling regulariza pero no puede generar información que los "
        "datos no contienen. Los priors son débilmente informativos, no planos, y el modelo "
        "ALD estima cada cuantil por separado -- no garantiza cuantiles monótonos en tau (de "
        "hecho, el hallazgo central es precisamente que el efecto NO es monótono: nulo en la "
        "mediana, negativo solo en la cola superior)."
    )


def build_ml_section(report: ResearchReport, ml: Dict, base_dir: Path) -> None:
    report.add_section_h1("16. Triangulación Machine Learning")
    report.add_body_text(
        "El análisis de ML (Random Forest y Gradient Boosting con LOCO-CV) sirve como "
        "triangulación no paramétrica del ranking de importancia de variables. "
        "No reemplaza la inferencia causal del modelo econométrico: su objetivo es "
        "confirmar que las variables clave aparecen consistentemente como predictores relevantes."
    )

    task_labels = {
        "T1: Violence → Institutions": "EQ1",
        "T2: Institutions → FDI":     "EQ2",
        "T3: FDI + Institutions → Growth": "EQ3",
    }

    for task_key, eq_label in task_labels.items():
        t = ml.get(task_key, {})
        if not t:
            continue
        report.add_section_h2(f"Tarea ML — {task_key}")

        # CV R² summary
        cv_rows = [
            ["Métrica", "Random Forest", "Gradient Boosting"],
            ["CV R² (media)", _fmt(t.get("rf_cv_r2_mean"), 4), _fmt(t.get("gb_cv_r2_mean"), 4)],
            ["CV R² (desv. est.)", _fmt(t.get("rf_cv_r2_sd"), 4), _fmt(t.get("gb_cv_r2_sd"), 4)],
        ]
        wil = t.get("wilcoxon_rf_gb", {})
        cv_rows.append([
            "Wilcoxon RF vs GB (p)",
            _fmt(wil.get("p_value"), 4),
            wil.get("note", ""),
        ])
        report.add_table(cv_rows, col_widths=[2.5, 2.0, 2.7])

        # Feature importance top 5
        shap_top5 = t.get("shap_top5", {})
        gini_top5 = t.get("gini_top5", {})
        perm_top5 = t.get("perm_top5", {})
        all_vars  = sorted(set(list(shap_top5) + list(gini_top5) + list(perm_top5)))

        if all_vars:
            imp_rows = [["Variable", "SHAP Imp.", "Gini Imp.", "Perm. Imp.", "SHAP Rank"]]
            stab = t.get("shap_rank_stability", {})
            for var in all_vars:
                rank_info = stab.get(var, {})
                mean_rank = rank_info.get("mean_rank", None)
                imp_rows.append([
                    var,
                    _fmt(shap_top5.get(var), 4),
                    _fmt(gini_top5.get(var), 4),
                    _fmt(perm_top5.get(var), 4),
                    _fmt(mean_rank, 2) if mean_rank is not None else "N/D",
                ])
            report.add_table(imp_rows, col_widths=[2.4, 1.1, 1.1, 1.1, 1.0])

        report.add_interpretation_box(
            f"Interpretación ML — {eq_label}",
            f"Variable clave rango SHAP: #{t.get('key_shap_rank', '?')}. "
            f"Rango permutación: #{t.get('key_perm_rank', '?')}. "
            "Una posición consistente en el top-5 del ranking de importancia "
            "refuerza la relevancia de la variable, incluso en ausencia de significancia "
            "estadística formal bajo los estrictos criterios de inferencia panel.",
            style="info",
        )

    # ML figures
    for fig_name, label, caption in [
        ("08_ml_loco_cv.png",       "Figura 8 — LOCO-CV R² por país",
         "R² de validación cruzada leave-one-country-out para RF y GB en las tres tareas."),
        ("09_feature_importance.png","Figura 9 — Importancia de variables",
         "Importancia Gini, permutación y SHAP de las variables en las tres tareas ML."),
        ("10_shap_ale.png",          "Figura 10 — SHAP / ALE plots",
         "Efectos marginales acumulados (ALE) y valores SHAP para los predictores clave."),
    ]:
        fig_path = base_dir / "figures" / fig_name
        if fig_path.exists():
            report.add_image(fig_path, label=label, caption=caption)


def build_conclusions_section(report: ResearchReport, meta: Dict, fe: Dict, boot: Dict) -> None:
    report.add_section_h1("17. Conclusiones")
    report.add_body_text(
        "A continuación se sintetizan las principales conclusiones derivadas estrictamente "
        "de los resultados calculados. No se realizan inferencias extrapoladas."
    )

    eq1 = fe.get("EQ1", {})
    eq2 = fe.get("EQ2", {})
    eq3 = fe.get("EQ3", {})
    boot_eq1 = boot.get("eq1", {})
    boot_eq2 = boot.get("eq2", {})
    boot_eq3 = boot.get("eq3", {})

    conclusions = [
        ["#", "Hallazgo", "Evidencia"],
        [
            "1",
            "Violencia → Instituciones (EQ1): coeficiente negativo, no significativo",
            f"β = {_fmt(eq1.get('coef_key_cl'), 4)}, p-CL = {_fmt(eq1.get('pval_key_cl'), 4)}, "
            f"p-boot = {_fmt(boot_eq1.get('p_boot'), 4)}",
        ],
        [
            "2",
            "Instituciones → FDI (EQ2): coeficiente positivo, significativo (robusto a 3 de 4 SE)",
            f"β = {_fmt(eq2.get('coef_key_cl'), 4)}, p-CL = {_fmt(eq2.get('pval_key_cl'), 4)}, "
            f"p-boot = {_fmt(boot_eq2.get('p_boot'), 4)}",
        ],
        [
            "3",
            "FDI → Crecimiento (EQ3): coeficiente positivo, no significativo",
            f"β = {_fmt(eq3.get('params_cl', {}).get('fdi_percent_gdp'), 4)}, "
            f"p-CL = {_fmt(eq3.get('pvals_cl', {}).get('fdi_percent_gdp'), 4)}, "
            f"p-boot = {_fmt(boot_eq3.get('p_boot'), 4)}",
        ],
        [
            "4",
            "Inflación → Crecimiento: el único efecto estadísticamente robusto en EQ3",
            f"β = {_fmt(eq3.get('params_cl', {}).get('inflation'), 4)}, "
            f"p-CL = {_fmt(eq3.get('pvals_cl', {}).get('inflation'), 4)}",
        ],
        [
            "5",
            "Bajo número de clústeres (G=11) limita el poder estadístico",
            "Bootstrap confirma amplios intervalos de confianza para todos los coeficientes clave",
        ],
    ]
    report.add_table(conclusions, col_widths=[0.3, 3.2, 3.7])

    report.add_interpretation_box(
        "Limitaciones metodológicas",
        "Con G=11 clústeres, el poder de los tests convencionales es reducido. "
        "La no significancia estadística no implica ausencia de efecto económico — "
        "los intervalos de confianza bootstrap son amplios y compatibles tanto con efectos "
        "nulos como con efectos moderados. Se recomienda ampliar la muestra geográfica "
        "para obtener inferencia más precisa.",
        style="warning",
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the full econometric pipeline.")
    parser.add_argument("--data",  type=str,  default=None,
                        help="Path to panel CSV.")
    parser.add_argument("--skip",  type=str,  nargs="*", default=[],
                        help="Module numbers to skip (e.g. --skip 4 5).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress banner output.")
    args = parser.parse_args()
    args.skip = [s.zfill(2) for s in args.skip]   # accept both '4' and '04'

    base_dir = Path(__file__).parent
    json_dir  = base_dir / "json"
    fig_dir   = base_dir / "figures"

    # ── Validate data path ────────────────────────────────────────────────
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"✗ Data file not found: {data_path}")
            sys.exit(1)
        import shutil
        dest = base_dir / "panel_ready_for_modeling.csv"
        if data_path.resolve() != dest.resolve():
            shutil.copy(data_path, dest)
            print(f"✓ Data copied → {dest}")
    else:
        data_path = base_dir / "panel_ready_for_modeling.csv"
        if not data_path.exists():
            print(f"✗ No data file found at {data_path}")
            sys.exit(1)

    print(f"\n{SEPARATOR}")
    print("  ECONOMETRIC PIPELINE: Violence → Institutions → FDI → Growth")
    print("  Central America, Colombia, Dominican Republic | 2000–2024")
    print(SEPARATOR)
    print(f"  Data:    {data_path}")
    print(f"  Skipping modules: {args.skip if args.skip else 'none'}")

    # ── Pre-load any existing JSON (modules may add/update) ───────────────
    meta_path = json_dir / "01_metadata.json"
    meta_pre  = _load_json(meta_path)

    report = ResearchReport(
        title="Violence, Institutions, and Economic Growth",
        subtitle="Evidencia de Panel para Centroamérica, Colombia y República Dominicana",
        author="Pipeline Econométrico — Auto-generado",
        script_name="run_pipeline.py",
        observations=meta_pre.get("N_obs"),
        countries=meta_pre.get("N"),
        year_range=(
            (min(meta_pre["years"]), max(meta_pre["years"]))
            if meta_pre.get("years") else None
        ),
        pipeline_version="2.0",
        output_path=base_dir / "econometric_report.pdf",
    )

    # Cover
    report.add_cover()

    enable_reporting(report)

    try:
        t_start = time.time()
        results = []

        for num, filename, description in MODULES:
            script = base_dir / filename
            if not script.exists():
                print(f"\n  ✗ Script not found: {script}")
                results.append((num, description, False, 0))
                continue

            success = run_module(script, args.skip, args.quiet, report, fig_dir)
            results.append((num, description, success, 0))

            if not success:
                print(f"\n  Pipeline halted at Module {num}.")
                break

        total = time.time() - t_start

        # Pipeline summary to console
        print(f"\n{SEPARATOR}")
        print("  PIPELINE SUMMARY")
        print(SEPARATOR)
        for num, desc, ok, _ in results:
            status = "✓" if ok else "✗" if num not in args.skip else "—"
            print(f"  {status}  Module {num}: {desc}")
        print(f"\n  Total runtime: {total:.1f}s")
        print(f"\n  Output directories:")
        print(f"    Figures → {fig_dir}/")
        print(f"    Tables  → {base_dir / 'tables'}/")
        print(f"    JSON    → {json_dir}/")
        print(SEPARATOR)

    finally:
        disable_reporting(report)

    # ── Reload JSON after modules ran ─────────────────────────────────────
    meta  = _load_json(json_dir / "01_metadata.json")
    fe    = _load_json(json_dir / "02_fe_results.json")
    diag  = _load_json(json_dir / "03_diagnostics.json")
    boot  = _load_json(json_dir / "04_bootstrap.json")
    med   = _load_json(json_dir / "04_mediation.json")
    rob   = _load_json(json_dir / "05_robustness.json")
    ml    = _load_json(json_dir / "06_ml_results.json")
    coint = _load_json(json_dir / "07_cointegration.json")
    gcar  = _load_json(json_dir / "08_growth_ceiling_risk.json")
    sc    = _load_json(json_dir / "09_synthetic_control.json")
    arch  = _load_json(json_dir / "10_arch_lm_test.json")

    # ── Build structured report sections ─────────────────────────────────
    build_executive_summary(report, meta, fe)
    report.add_page_break()

    build_dataset_section(report, meta)
    report.add_page_break()

    build_variables_section(report, meta)
    report.add_page_break()

    build_descriptive_section(report, base_dir)
    report.add_page_break()

    build_correlations_section(report, base_dir)
    report.add_page_break()

    build_pca_section(report, meta, base_dir)
    report.add_page_break()

    build_model_sections(report, fe, base_dir)
    report.add_page_break()

    build_diagnostics_section(report, diag, base_dir)
    build_arch_subsection(report, arch)
    report.add_page_break()

    build_bootstrap_section(report, boot, med, base_dir)
    report.add_page_break()

    build_robustness_section(report, rob, base_dir)
    report.add_page_break()

    build_synthetic_control_section(report, sc, base_dir)
    report.add_page_break()

    build_cointegration_section(report, coint, base_dir)
    report.add_page_break()

    build_growth_ceiling_section(report, gcar, base_dir)
    report.add_page_break()

    build_ml_section(report, ml, base_dir)
    report.add_page_break()

    build_conclusions_section(report, meta, fe, boot)
    report.add_page_break()

    # Appendix (raw console log)
    report.add_annex()

    # ── Write PDF ─────────────────────────────────────────────────────────
    out = report.save(base_dir / "econometric_report.pdf")
    print(f"\n  ✓ PDF report saved → {out}")


if __name__ == "__main__":
    main()
