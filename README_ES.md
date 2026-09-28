# Violencia, Instituciones y Crecimiento en Centroamérica y el Caribe

Este repositorio implementa un flujo de trabajo empírico para estudiar cómo el crimen violento afecta la calidad institucional y, a través de ese canal, el crecimiento económico. El proyecto combina un panel de datos econométrico estructurado con técnicas de aprendizaje automático para examinarse si la relación entre la exposición a homicidios y el crecimiento es directa o está mediada por la calidad institucional y la inversión extranjera directa (IED). El análisis se centra en un panel de países de Centroamérica, Colombia y la República Dominicana durante el período 2000-2024.

## Pregunta de Investigación

¿Cómo afecta la violencia, medida por las tasas de homicidio, a la calidad institucional y al desempeño económico posterior, y está la relación entre el crimen y el crecimiento mediada por las instituciones y la IED?

## Metodología

El proyecto utiliza un diseño de panel longitudinal con efectos fijos por país y año. La especificación de referencia es un MCO agrupado, seguido de modelos de efectos fijos bidireccionales que absorben la heterogeneidad invariante en el tiempo por país y los choques comunes. La inferencia se informa con errores estándar agrupados convencionales y con alternativas más conservadoras, incluyendo errores estándar de Driscoll-Kraay para considerar la dependencia transversal, un estimador CR2 (Bell-McCaffrey) con grados de libertad de Satterthwaite, e inferencia de bootstrap de grupo silvestre (wild cluster bootstrap) para paneles con pocos grupos. Dado que la cadena de tres ecuaciones (violencia → instituciones → IED → crecimiento) se estima como modelos independientes para evitar un problema de regresores generados, el mecanismo hipotetizado se pone a prueba adicionalmente como sistema: un bootstrap por clúster de país del producto de los tres coeficientes de la cadena ofrece una prueba formal del efecto indirecto, en lugar de apoyarse solo en la significancia de cada eslabón por separado. Las comprobaciones de robustez incluyen exclusión de cada país (Leave-One-Country-Out), estructuras de rezago alternativas para los tres eslabones (no solo violencia→instituciones), una verificación de control post-tratamiento, tendencias lineales específicas por país y una prueba de raíz unitaria de panel (Fisher-ADF). Además, se utiliza un análisis de componentes principales (PCA) para construir un índice institucional de robustez a partir de los Indicadores de Gobernanza Mundial (WGI), y se emplean métodos de aprendizaje automático como una capa de triangulación exploratoria en lugar de un sustituto para la estimación causal, con una reconstrucción del índice institucional libre de fuga de datos (fold-safe) bajo validación cruzada Leave-One-Country-Out.

El flujo de trabajo se ejecuta en ocho etapas:
1.  **Carga de datos**: lee un conjunto de datos de panel limpio que contiene observaciones país-año.
2.  **Limpieza y ETL**: valida los datos, evalúa los valores perdidos y estandariza y combina las variables institucionales en un índice.
3.  **Ingeniería de características**: enriquece el panel con tendencias temporales, variables de violencia rezagadas y medidas institucionales derivadas.
4.  **Construcción de PCA**: crea un índice institucional de robustez utilizando PCA en los indicadores de gobernanza.
5.  **Estimación econométrica**: estima modelos de efectos fijos bidireccionales para las relaciones violencia-instituciones, instituciones-IED y IED+instituciones-crecimiento, cada una reestimada también bajo una alternativa con rezago de un año.
6.  **Triangulación con aprendizaje automático**: entrena modelos de bosque aleatorio y aumento de gradiente para una comparación exploratoria con las estimaciones econométricas, usando validación cruzada Leave-One-Country-Out con un índice institucional libre de fuga de datos.
7.  **Validación**: ejecuta diagnósticos, inferencia de bootstrap (incluyendo un bootstrap por clúster del efecto indirecto de la cadena completa de mediación) y comprobaciones de robustez para evaluar la sensibilidad a observaciones influyentes, opciones de rezago, conjuntos de controles, tendencias específicas por país, no estacionariedad y especificación del modelo.
8.  **Cointegración de panel y corrección de errores**: para los pares de variables no estacionarias detectadas en la etapa 7 (homicidios, instituciones, PIB per cápita), un test de dos pasos Engle-Granger/Kao verifica si comparten una relación de equilibrio de largo plazo genuina antes de estimar un modelo de corrección de errores; si no la comparten, esto se reporta como un hallazgo en sí mismo en vez de asumirse silenciosamente.

## Revisión Metodológica (Septiembre 2026)

Una revisión metodológica del pipeline —realizada de cara a una eventual publicación en arXiv o presentación universitaria— identificó y corrigió varios problemas comunes en estudios de panel con N pequeño. El detalle técnico completo vive en los docstrings de cada módulo y en el historial de commits (`79a37a6` a `7563671`):

1. **La sensibilidad al estimador de errores estándar ahora se reporta en vez de citar el número más favorable.** El coeficiente instituciones → IED (EQ2) parece "marginalmente significativo" (p ≈ 0.055) solo bajo errores clusterizados convencionales —el menos conservador de los cuatro estimadores que calcula el pipeline, y conocido por sobre-rechazar con G=8 clústeres. Bajo Driscoll-Kraay (p ≈ 0.12) y CR2 Bell-McCaffrey (p ≈ 0.20) no es significativo al 10%; el bootstrap de clúster silvestre da p ≈ 0.06. Ver `tables/se_comparison.tex` y la Sección 11 del reporte PDF.
2. **La cadena de transmisión completa ahora se prueba formalmente, no solo se narra.** El Módulo 04 hace bootstrap del producto de los tres coeficientes de la cadena (β₁·γ₁·δ₁) remuestreando países. Estimación puntual: −0.017, IC 95% [−0.079, 0.065], p = 0.62 — la cadena como sistema no es distinguible de cero, aunque el signo de cada eslabón coincida con la teoría.
3. **La estructura de rezagos ahora se prueba en EQ2 y EQ3, no solo en EQ1.** El modelo teórico argumenta que instituciones y crecimiento se ajustan más lento que la violencia, pero las especificaciones primarias de EQ2/EQ3 usaban regresores contemporáneos. Reestimar con rezagos de un año (Módulo 05) invierte el signo de EQ3 (+0.137 contemporáneo vs. −0.121 rezagado).
4. **Se agregó una verificación de control post-tratamiento ("bad control").** `gdp_per_capita_log` es plausiblemente posterior a instituciones/IED, no un confusor exógeno; excluirlo mueve el coeficiente de EQ2 de 0.900 (p=0.055) a 0.739 (p=0.128).
5. **Se agregaron tendencias lineales específicas por país y una prueba de raíz unitaria de panel (Fisher-ADF / Maddala-Wu).** Los efectos fijos bidireccionales no eliminan tendencias específicas por país, y `homicide_rate_log`, `inst_avg` y `gdp_per_capita_log` no logran rechazar raíz unitaria en este panel de 25 años — consistente con que EQ1 (violencia → instituciones) sea el eslabón menos estable en todas las comprobaciones del pipeline.
6. **La validación cruzada Leave-One-Country-Out ya no filtra el país excluido hacia el índice institucional.** `inst_avg` se reconstruye dentro de cada pliegue usando solo las constantes de escala de los países de entrenamiento.
7. **Se corrigió un bug silencioso en la tabla de convergencia FE-ML** — siempre imprimía "?" por una búsqueda incorrecta de clave JSON para los coeficientes de EQ3.

## Corrección de Origen de Datos (Septiembre 2026)

Una auditoría posterior de la cadena de extracción (`data_extraction/`, `archive/legacy_scripts/`) —motivada por la sospecha de que mezclar escalas distintas causaba problemas— encontró que las escalas en sí no eran el problema (mezclar percentiles 0-100, % del PIB y logs en la misma regresión no sesga FE/OLS ni el ML basado en árboles), pero sí encontró 4 bugs reales de datos:

1. **`homicide_rate` se winsorizaba silenciosamente al percentil 1-99** antes del log. Con 179 observaciones, los 4 puntos recortados eran, los 4, El Salvador: su pico de violencia 2015/2016 (107.6→83.6, 85.1→83.6) y sus mínimos post-reforma 2023/2024 (2.24→5.43, 1.90→5.43) — exactamente el cambio abrupto que motiva esta investigación, aplanado antes de llegar a EQ1. Eliminado.
2. **`trade_percent_gdp` estaba 100% vacío** en todas las observaciones, por un bug que sumaba `imports_percent_gdp`, columna que nunca se extrajo. Esto explica por qué "Extended controls" siempre fallaba en el Módulo 05. Corregido agregando el indicador faltante del Banco Mundial.
3. **`gdp_per_capita` mezclaba dólares nominales y reales para Guatemala específicamente** — un script de relleno de huecos usaba el indicador equivocado para ese país. Corregido para usar el mismo indicador que los otros 7 países.
4. **El único script de extracción de WGI en vivo pedía la escala incorrecta** (el "estimate" en vez del "score" 0-100 que documenta todo el pipeline) — código muerto, ya que la fuente real era un archivo descargado manualmente y nunca versionado. Corregido, cerrando el hueco de reproducibilidad.

El panel ahora está balanceado a nivel país-año (200 filas = 8×25, antes 179). Al re-correr el pipeline completo con los datos corregidos, las conclusiones de EQ1, EQ3 y el test de mediación no cambian, pero **EQ2 (instituciones→IED) se vuelve mucho más robusto**: p=0.055/0.116/0.202/0.062 (clusterizado/Driscoll-Kraay/CR2/bootstrap) → **p=0.007/0.085/0.126/0.012**.

## Resultados Esperados

Dado el diseño del proyecto y las limitaciones de los datos (solo 8 países independientes), los resultados se interpretan como asociaciones sugestivas en lugar de efectos causales definitivos. Cada eslabón de la cadena presenta el signo esperado por la teoría:
- Una asociación negativa entre las tasas rezagadas de homicidio y la calidad institucional (no significativa bajo ningún estimador de varianza).
- Una asociación positiva entre la calidad institucional y la IED (con los datos corregidos, significativa al 5% bajo tres de los cuatro estimadores: clusterizado p=0.007, bootstrap p=0.012, Driscoll-Kraay p=0.085; CR2 p=0.126 — ver Corrección de Origen de Datos arriba).
- Una asociación positiva entre la IED y el crecimiento económico (no significativa; se invierte de signo bajo especificación con rezagos).

Sin embargo, un bootstrap formal por clúster de país del efecto indirecto conjunto (el producto de los tres coeficientes) **no rechaza la hipótesis nula de que la cadena completa sea cero** (IC 95% = [−0.069, 0.051], p=0.65). Dada la pequeña cantidad de grupos (N=8), las preocupaciones potenciales de endogeneidad, y esta prueba formal de mediación, estos resultados deben presentarse como patrones consistentes con el marco teórico —con EQ2 ahora en base empírica sólida—, no como pruebas de causalidad ni como un mecanismo estadísticamente establecido en su conjunto.

## Limitaciones

- **Número pequeño de grupos (N=8)**: Todos los métodos de inferencia (incluso el bootstrap de grupo silvestre y los errores estándar de Driscoll-Kraay) dependen de propiedades asintóticas que pueden no ser válidas con tan pocos grupos independientes. Los intervalos de confianza deben interpretarse como sugestivos.
- **Endogeneidad potencial**: Aunque se utilizan tasas de homicidio rezagadas para abordar la simultaneidad inversa, las variables omitidas (por ejemplo, capacidad estatal, disturbance social) podrían afectar tanto la violencia como las instituciones.
- **Error de medición en las instituciones**: Los WGI son indicadores basados en percepciones y pueden contener error de medición.
- **Canal secuencial asumido**: El modelo asume un camino estricto violencia → instituciones → IED → crecimiento, pero podrían existir bucles de retroalimentación o efectos simultáneos. Un bootstrap formal del efecto indirecto conjunto (Revisión Metodológica, punto 2) confirma que esta cadena no supera un test de significancia como sistema con el tamaño muestral disponible.
- **Selección de variables de control**: Algunas variables de control tienen poca variación dentro de los países (por ejemplo, comercio como % del PIB), lo que limita lo que los efectos fijos pueden identificar. `gdp_per_capita_log` en particular es un control potencialmente post-tratamiento (Revisión Metodológica, punto 4).
- **No estacionariedad**: `homicide_rate_log`, `inst_avg` y `gdp_per_capita_log` no rechazan raíz unitaria en un test de panel Fisher-ADF, un riesgo de regresión espuria que los efectos fijos bidireccionales no corrigen por sí solos (Revisión Metodológica, punto 5).
- **Sin evidencia de cointegración (Módulo 07)**: dado que homicidios, instituciones y PIB per cápita son I(1), se probó formalmente si comparten una relación de equilibrio de largo plazo (Engle-Granger/Kao de dos pasos). Ningún par mostró evidencia de cointegración, lo que ofrece una explicación estructural adicional —más allá del bajo poder por G=8— de por qué EQ1 (violencia→instituciones) es el eslabón más inestable en todas las pruebas de este pipeline.

## Contribución

A pesar de las limitaciones, este proyecto contribuye al proporcionar:
- Un flujo de trabajo transparente y reproducible para el análisis de panel de datos en un contexto de recursos limitados.
- Una demostración de cómo abordar la inferencia en paneles con muy pocos grupos (énfasis en el bootstrap de grupo silvestre).
- Un marco para explorar la violencia-instituciones-crecimiento nexo en una región subestudiada utilizando datos accesibles.
- Una base para futuros trabajos que puedan incorporar variables instrumentales, datos de mayor frecuencia o métodos estructurales cuando los datos lo permitan.

## Cómo Ejecutar el Proyecto

1.  **Configuración del entorno**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # En Linux/macOS
    .venv\Scripts\activate      # En Windows PowerShell
    pip install -r requirements.txt
    ```
2.  **Ejecutar el pipeline principal**:
    Desde la raíz del repositorio:
    ```bash
    python econometric_pipeline/pipeline/run_pipeline.py
    ```
    Opcionalmente, se pueden omitir ciertos módulos (por ejemplo, para omitir el bootstrap y la triangulación de ML):
    ```bash
    python econometric_pipeline/pipeline/run_pipeline.py --skip 4 5 6
    ```
3.  **Salidas esperadas**:
    - Figuras en `econometric_pipeline/pipeline/figures`
    - Tablas en `econometric_pipeline/pipeline/tables`
    - Resúmenes JSON en `econometric_pipeline/pipeline/json`
    - Un informe final en PDF en `econometric_pipeline/pipeline/econometric_report.pdf`

## Estructura del Repositorio

```
data_analyzer/
├── econometric_pipeline/
│   └── pipeline/
│       ├── 01_data_preparation.py
│       ├── 02_panel_estimation.py
│       ├── 03_diagnostics.py
│       ├── 04_bootstrap_inference.py
│       ├── 05_robustness.py
│       ├── 06_ml_triangulation.py
│       ├── run_pipeline.py
│       ├── utils.py
│       ├── research_report.py
│       ├── figures/
│       ├── json/
│       └── tables/
├── archive/                 # Scripts y datos heredados (exploratorios)
├── data_extraction/         # Scripts auxiliares para la extracción de datos (no necesarios para el pipeline principal)
├── documentation/
├── foundational_datasets/
└── requirements.txt
```

## Notas para el Usuario

- El flujo de trabajo reproducible se centra en el pipeline bajo `econometric_pipeline/pipeline`. Los scripts exploratorios heredados y los conjuntos de datos redundantes se han movido a `archive/` para mantener el enfoque en el análisis principal.
- Los conjuntos de datos esenciales para la reproducibilidad son la entrada del pipeline (`panel_ready_for_modeling.csv`) y los resultados generados en el directorio del pipeline.
- El proyecto está diseñado para ser lo más transparente posible, con una separación clara entre el flujo de trabajo central y los scripts exploratorios antiguos.

## Contacto

Para preguntas sobre el proyecto, por favor refiérase al archivo `README.md` en inglés para obtener detalles técnicos adicionales o contacte al mantenedor del repositorio.