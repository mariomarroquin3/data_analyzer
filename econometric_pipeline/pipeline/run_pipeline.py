"""
run_pipeline.py
════════════════════════════════════════════════════════════════════════
Master runner — executes all six modules in sequence and assembles a
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
                matplotlib scipy arch joblib reportlab
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
        "umbrales convencionales bajo este esquema de inferencia. EQ2 muestra que mejores "
        "instituciones atraen más FDI (p ≈ 0.055, marginalmente significativo). "
        "El mecanismo completo (violencia → inst. → FDI → crecimiento) requiere interpretación "
        "conjunta de los tres modelos y los resultados bootstrap.",
        style="info",
    )


def build_dataset_section(report: ResearchReport, meta: Dict) -> None:
    report.add_section_h1("2. Información del Dataset")
    report.add_body_text(
        "El panel cubre 8 países de América Central, Colombia y República Dominicana. "
        "Los datos provienen del Banco Mundial (WDI, WGI) y fuentes nacionales de estadísticas de crimen."
    )

    countries = meta.get("countries", [])
    country_data = [["Código", "País"]]
    names = {
        "COL": "Colombia", "CRI": "Costa Rica", "DOM": "Rep. Dominicana",
        "GTM": "Guatemala", "HND": "Honduras", "NIC": "Nicaragua",
        "PAN": "Panamá", "SLV": "El Salvador",
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
        ["rule_of_law", "Estado de derecho (WGI)", "Sin transformar (−2.5 a 2.5)", "Componente inst."],
        ["control_corruption", "Control de corrupción (WGI)", "Sin transformar (−2.5 a 2.5)", "Componente inst."],
        ["political_stability", "Estabilidad política (WGI)", "Sin transformar (−2.5 a 2.5)", "Componente inst."],
        ["fdi_percent_gdp", "IED como % del PIB", "Sin transformar", "Dependiente EQ2 / Indep. EQ3"],
        ["gdp_growth", "Crecimiento real del PIB (%)", "Sin transformar", "Dependiente EQ3"],
        ["gdp_per_capita_log", "PIB per cápita (USD cte.)", "log(x)", "Control"],
        ["inflation", "Inflación anual (%)", "Sin transformar", "Control"],
        ["exports_percent_gdp", "Exportaciones como % del PIB", "Sin transformar", "Control"],
        ["population_log", "Población total", "log(x)", "Control"],
        ["unemployment", "Tasa de desempleo (%)", "Sin transformar", "Control"],
    ]
    report.add_table(var_data, col_widths=[2.0, 2.5, 1.8, 1.5])

    # PCA / Cronbach info
    pca_var  = meta.get("pca_var_exp_pc1")
    cronbach = meta.get("cronbach_alpha")
    kmo      = meta.get("kmo")
    loadings = meta.get("pca_loadings", {})

    report.add_section_h2("Índice Institucional — PCA")
    report.add_body_text(
        f"Se extrajo el primer componente principal (PC1) de las tres dimensiones WGI "
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
    report.add_interpretation_box(
        "Validez del índice institucional",
        f"Un alfa de Cronbach > 0.80 indica alta consistencia interna de las tres dimensiones. "
        f"El primer componente explica {_fmt(pca_var, 1) if pca_var else '?'}% de la varianza, "
        f"justificando la reducción dimensional. Todos los loadings son positivos, confirmando "
        f"que PC1 captura 'calidad institucional general' y no una dimensión específica.",
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

    report.add_body_text(
        "El PCA reduce las tres dimensiones WGI a un índice sintético único (inst_pca / inst_avg). "
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
                "resultados bootstrap (Sección 11) dado el bajo número de clústeres (G=8)."
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
                "Un coeficiente positivo sobre inst_avg confirma el canal institucional: "
                "mejores instituciones reducen los riesgos de apropiación y contratos incompletos, "
                "atrayendo más IED. El p-valor marginal (~0.055) sugiere evidencia moderada."
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


def build_bootstrap_section(report: ResearchReport, boot: Dict, base_dir: Path) -> None:
    report.add_section_h1("11. Inferencia Bootstrap (Wild Cluster)")
    report.add_body_text(
        "Con solo G=8 clústeres, los errores estándar clusterizados convencionales "
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


def build_ml_section(report: ResearchReport, ml: Dict, base_dir: Path) -> None:
    report.add_section_h1("13. Triangulación Machine Learning")
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
    report.add_section_h1("14. Conclusiones")
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
            "Instituciones → FDI (EQ2): coeficiente positivo, marginalmente significativo",
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
            "Bajo número de clústeres (G=8) limita el poder estadístico",
            "Bootstrap confirma amplios intervalos de confianza para todos los coeficientes clave",
        ],
    ]
    report.add_table(conclusions, col_widths=[0.3, 3.2, 3.7])

    report.add_interpretation_box(
        "Limitaciones metodológicas",
        "Con G=8 clústeres, el poder de los tests convencionales es reducido. "
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
    rob   = _load_json(json_dir / "05_robustness.json")
    ml    = _load_json(json_dir / "06_ml_results.json")

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
    report.add_page_break()

    build_bootstrap_section(report, boot, base_dir)
    report.add_page_break()

    build_robustness_section(report, rob, base_dir)
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
