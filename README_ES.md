# Violencia, Instituciones y Crecimiento en América Latina y el Caribe

Este repositorio implementa un flujo de trabajo empírico para estudiar cómo el crimen violento afecta la calidad institucional y, a través de ese canal, el crecimiento económico. El proyecto combina un panel de datos econométrico estructurado con técnicas de aprendizaje automático para examinarse si la relación entre la exposición a homicidios y el crecimiento es directa o está mediada por la calidad institucional y la inversión extranjera directa (IED). El análisis se centra en un panel de 11 países —los 8 originales de Centroamérica, Colombia y la República Dominicana, más México, Ecuador y Perú (agregados en septiembre de 2026; ver Extensión del Panel más abajo)— durante el período 2000-2024.

## Pregunta de Investigación

¿Cómo afecta la violencia, medida por las tasas de homicidio, a la calidad institucional y al desempeño económico posterior, y está la relación entre el crimen y el crecimiento mediada por las instituciones y la IED?

## Metodología

El proyecto utiliza un diseño de panel longitudinal con efectos fijos por país y año. La especificación de referencia es un MCO agrupado, seguido de modelos de efectos fijos bidireccionales que absorben la heterogeneidad invariante en el tiempo por país y los choques comunes. La inferencia se informa con errores estándar agrupados convencionales y con alternativas más conservadoras, incluyendo errores estándar de Driscoll-Kraay para considerar la dependencia transversal, un estimador CR2 (Bell-McCaffrey) con grados de libertad de Satterthwaite, e inferencia de bootstrap de grupo silvestre (wild cluster bootstrap) para paneles con pocos grupos. Dado que la cadena de tres ecuaciones (violencia → instituciones → IED → crecimiento) se estima como modelos independientes para evitar un problema de regresores generados, el mecanismo hipotetizado se pone a prueba adicionalmente como sistema: un bootstrap por clúster de país del producto de los tres coeficientes de la cadena ofrece una prueba formal del efecto indirecto, en lugar de apoyarse solo en la significancia de cada eslabón por separado. Las comprobaciones de robustez incluyen exclusión de cada país (Leave-One-Country-Out), estructuras de rezago alternativas para los tres eslabones (no solo violencia→instituciones), una verificación de control post-tratamiento, tendencias lineales específicas por país y una prueba de raíz unitaria de panel (Fisher-ADF). Además, se utiliza un análisis de componentes principales (PCA) para construir un índice institucional de robustez a partir de los Indicadores de Gobernanza Mundial (WGI), y se emplean métodos de aprendizaje automático como una capa de triangulación exploratoria en lugar de un sustituto para la estimación causal, con una reconstrucción del índice institucional libre de fuga de datos (fold-safe) bajo validación cruzada Leave-One-Country-Out. Finalmente, un hallazgo exploratorio de regresión cuantílica —que la violencia limita la cola superior del crecimiento del PIB sin desplazar su mediana— se re-estima como un modelo bayesiano jerárquico (MCMC, verosimilitud Asimétrica de Laplace) con partial pooling entre países, reportado con incertidumbre posterior completa en vez de errores estándar asintóticos de regresión cuantílica.

El flujo de trabajo se ejecuta en nueve etapas:
1.  **Carga de datos**: lee un conjunto de datos de panel limpio que contiene observaciones país-año.
2.  **Limpieza y ETL**: valida los datos, evalúa los valores perdidos y estandariza y combina las variables institucionales en un índice.
3.  **Ingeniería de características**: enriquece el panel con tendencias temporales, variables de violencia rezagadas y medidas institucionales derivadas.
4.  **Construcción de PCA**: crea un índice institucional de robustez utilizando PCA en los indicadores de gobernanza.
5.  **Estimación econométrica**: estima modelos de efectos fijos bidireccionales para las relaciones violencia-instituciones, instituciones-IED y IED+instituciones-crecimiento, cada una reestimada también bajo una alternativa con rezago de un año.
6.  **Triangulación con aprendizaje automático**: entrena modelos de bosque aleatorio y aumento de gradiente para una comparación exploratoria con las estimaciones econométricas, usando validación cruzada Leave-One-Country-Out con un índice institucional libre de fuga de datos.
7.  **Validación**: ejecuta diagnósticos, inferencia de bootstrap (incluyendo un bootstrap por clúster del efecto indirecto de la cadena completa de mediación) y comprobaciones de robustez para evaluar la sensibilidad a observaciones influyentes, opciones de rezago, conjuntos de controles, tendencias específicas por país, no estacionariedad y especificación del modelo.
8.  **Cointegración de panel y corrección de errores**: para los pares de variables no estacionarias detectadas en la etapa 7 (homicidios, instituciones, PIB per cápita), un test de dos pasos Engle-Granger/Kao verifica si comparten una relación de equilibrio de largo plazo genuina antes de estimar un modelo de corrección de errores; si no la comparten, esto se reporta como un hallazgo en sí mismo en vez de asumirse silenciosamente.
9.  **Growth-Ceiling-at-Risk (MCMC bayesiano)**: una regresión cuantílica jerárquica bayesiana (verosimilitud Asimétrica de Laplace, partial pooling entre países) re-estima un hallazgo exploratorio según el cual la violencia limita la cola superior del crecimiento del PIB sin afectar su mediana, sustituyendo las variables dummy por país (LSDV) y la inferencia asintótica de regresión cuantílica por regularización de partial pooling e incertidumbre posterior completa.

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

## Extensión del Panel (Septiembre 2026)

Todas las advertencias sobre G=8 en este documento especulaban que el número pequeño de países era la restricción estadística que más pesaba en los resultados. Esa hipótesis ya se puso a prueba directamente: se agregaron México, Ecuador y Perú al panel (11 países, 274 filas país-año, antes 200), incluyendo específicamente a México por su peso regional en dinámicas de seguridad (violencia asociada al narcotráfico). Los mismos scripts de extracción se reutilizaron sin cambios de lógica —solo creció la lista de países— confirmando que el pipeline de datos corregido en la fase anterior generaliza bien.

El efecto sobre los resultados fue considerable, en la dirección que predecían las advertencias de G pequeño:

| | EQ1 (violencia→instituciones) | EQ2 (instituciones→IED) | EQ3 (IED→crecimiento) |
|---|---|---|---|
| p (clusterizado), 8→11 países | 0.352 → **0.093*** | 0.007 → **0.0005*** | 0.306 → **0.010*** |
| p (Driscoll-Kraay) | 0.417 → **0.010*** | 0.085 → **0.003*** | — |
| p (bootstrap de clúster silvestre) | 0.290 → **0.085*** | 0.012 → **0.002*** | 0.450 → 0.108 |

El bootstrap de mediación del efecto indirecto conjunto pasó de una estimación puntual de -0.0075 (IC 95% [−0.069, 0.051], p=0.65) a -0.031 (IC 95% [−0.137, 0.015], p=0.144) —el IC todavía incluye cero, pero la cadena está bastante más cerca de la significancia formal como sistema que con N=8. El test de cointegración del Módulo 07 también cambió cualitativamente: el par homicidios/PIB per cápita (T3) ahora muestra evidencia de cointegración (Fisher-ADF sobre residuos, p=0.011) donde antes no la había, y el modelo de corrección de errores resultante encuentra una velocidad de ajuste significativa (phi=-0.109, p<0.01): el PIB per cápita corrige cerca del 11% de cualquier desviación de su relación de largo plazo con la violencia cada año — el primer resultado dinámico (no solo estático) de este proyecto.

## Resultados Esperados

Dado el diseño del proyecto, los resultados se interpretan como asociaciones sugestivas en lugar de efectos causales definitivos. Los resultados dependen de forma material de dos decisiones de especificación que este documento reporta con su cifra antes/después, no solo la especificación final: el tamaño del panel (8 → 11 países, ver Extensión del Panel) y la construcción del índice institucional (3 → 6 dimensiones WGI, ver Extensión del Índice Institucional). **Bajo la especificación actual** (11 países, índice de 6 dimensiones):
- La asociación entre violencia rezagada y calidad institucional **ya no es significativa bajo ningún estimador** (p=0.462 clusterizado, p=0.310 Driscoll-Kraay, p=0.562 bootstrap) — era marginalmente significativa con el índice de 3 dimensiones, pero no sobrevive una construcción más completa y estándar de la misma medida.
- La asociación entre calidad institucional e IED sigue siendo significativa en su especificación primaria (el eslabón más defendible: p=0.030 clusterizado, p=0.006 Driscoll-Kraay, p=0.038 bootstrap), pero **ya no sobrevive** el rezago de un año (p=0.318) ni las tendencias específicas por país (p=0.260) que sí superaba con el índice de 3 dimensiones.
- La asociación entre IED y crecimiento económico sigue siendo significativa en contemporáneo (p=0.010; no depende de inst_avg) y sigue sin sobrevivir la especificación con rezagos.

Un bootstrap formal por clúster de país del efecto indirecto conjunto (el producto de los tres coeficientes) **no rechaza la hipótesis nula al 5%** (IC 95% = [−0.113, 0.020], p=0.548) y está, si acaso, más lejos de hacerlo que con el índice de 3 dimensiones. La lectura honesta: el eslabón instituciones→IED es razonablemente sólido; el eslabón violencia→instituciones debe tratarse ahora como una hipótesis de trabajo, no como un resultado establecido; y la *cadena como sistema mediado único* sigue sin probarse.

## Hallazgo Exploratorio: Un Patrón de "Techo de Crecimiento" (Septiembre 2026)

Antes de comprometerse a construir un módulo completo de Growth-at-Risk (GaR), se hizo una revisión exploratoria rápida: una regresión cuantílica agrupada (Koenker) de `gdp_growth` sobre `homicide_rate_log_lag1`, con efectos fijos por país (LSDV) y una tendencia lineal por año, en los cuantiles 0.05–0.95 (N=231, 11 países). La teoría clásica de GaR (Adrian, Boyarchenko y Giannone, 2019, "Vulnerable Growth") predice que la violencia debería golpear con más fuerza la cola *inferior* de la distribución del crecimiento; este panel muestra lo contrario:

| Cuantil | coef(homicidios, rezago 1) | valor p |
|---|---|---|
| 0.05 | +2.68 | 0.030** |
| 0.10 | +0.65 | 0.561 |
| 0.25 | +0.18 | 0.690 |
| 0.50 (mediana) | -0.10 | 0.813 |
| 0.75 | -0.37 | 0.443 |
| 0.90 | **-2.45** | **0.0001***|
| 0.95 | **-2.06** | **0.0069***|

La violencia prácticamente no tiene efecto cerca de la mediana, pero sí un efecto *negativo*, grande y estadísticamente significativo sobre la cola *superior* (q=0.90/0.95): un patrón de "techo de crecimiento", donde la violencia alta no empeora los años malos, sino que limita cuánto puede crecer un año bueno. El mismo chequeo usando `inst_avg` (instituciones) en lugar de violencia como variable condicionante no encontró nada en ningún cuantil (todo p>0.27) — el efecto de techo parece específico a la violencia, no a la calidad institucional en general.

Como un solo país influyente podría producir fácilmente este tipo de resultado de cola con solo 11 clústeres, los coeficientes de la cola superior se sometieron a un chequeo Leave-One-Country-Out antes de tratar el patrón como real: reajustando q=0.90 y q=0.95 excluyendo cada uno de los 11 países, uno a la vez. El coeficiente se mantuvo negativo y significativo en **los 11** ajustes leave-one-out en ambos cuantiles (q=0.90, rango: -1.62 a -3.21, todo p<0.03; q=0.95, rango: -1.69 a -3.07, el caso más débil p=0.087 excluyendo El Salvador) — sin cambios de signo, y excluir a México si acaso fortalece el efecto. El patrón no es un artefacto de ningún país individual.

En su momento esto fue una revisión rápida, no un módulo comprometido al pipeline —pero suficientemente robusta como para motivar una extensión propiamente dicha de **Growth-Ceiling-at-Risk**. Ya está implementada como el Módulo 08; ver la sección inmediatamente debajo.

## Módulo Growth-Ceiling-at-Risk (Septiembre 2026)

El patrón exploratorio anterior se re-estimó como [Módulo 08](econometric_pipeline/pipeline/08_growth_ceiling_risk.py): una regresión cuantílica jerárquica bayesiana (verosimilitud Asimétrica de Laplace, MCMC vía PyMC/NUTS) en lugar de variables dummy por país (LSDV) y errores estándar asintóticos de regresión cuantílica. El partial pooling encoge el intercepto de cada país hacia la media global en una magnitud que determinan los propios datos —una regularización que LSDV no puede ofrecer con solo G=11 clústeres— y cada cantidad se reporta como una distribución posterior completa (media, intervalo de mayor densidad al 94%, y P(coeficiente < 0 | datos)) en vez de un valor p basado en una aproximación asintótica ya señalada como poco confiable a este tamaño muestral en el resto de este pipeline.

| Cuantil | Coef. frecuentista (LSDV) | Media posterior bayesiana | 94% HDI | P(β<0 \| datos) |
|---|---|---|---|---|
| 0.05 | +2.21 (p=0.040**) | +1.31 | [-0.03, 2.53] | 0.027 |
| 0.10 | +0.25 (p=0.819) | +0.49 | [-0.53, 1.44] | 0.171 |
| 0.25 | +0.76 (p=0.045**) | +0.19 | [-0.46, 0.84] | 0.294 |
| 0.50 (mediana) | -0.08 (p=0.827) | -0.23 | [-0.77, 0.37] | 0.790 |
| 0.75 | -0.20 (p=0.650) | -0.45 | [-1.21, 0.26] | 0.880 |
| 0.90 | -2.23 (p=0.0005***) | **-1.76** | **[-2.93, -0.59]** | **0.997** |
| 0.95 | -2.06 (p=0.0043***) | **-2.08** | **[-3.02, -1.11]** | **1.000** |

La posterior bayesiana confirma el patrón frecuentista en todos los cuantiles —el encogimiento por partial pooling mueve las estimaciones puntuales de forma moderada (p. ej. q=0.90: -2.23 → -1.76) sin cambiar la conclusión cualitativa— y los diagnósticos de MCMC fueron limpios en todos los cuantiles (R-hat ≤ 1.004, tamaño de muestra efectivo > 1,100, cero transiciones divergentes en 4 cadenas × 1,000 muestras post-calentamiento cada una). Un chequeo secundario repitiendo el mismo modelo con `inst_avg` en lugar de violencia en q=0.90/0.95 no encontró efecto de techo (P(β<0|datos) = 0.367 y 0.597 —prácticamente un volado—, HDIs amplios y centrados cerca de cero) — el efecto es específico a la violencia, no una característica genérica de cualquier regresor en este panel. (Cifras con el índice institucional de 6 dimensiones; ver Extensión del Índice Institucional más abajo.)

**Escenario Growth-Ceiling-at-Risk.** Manteniendo país y año en su nivel promedio, la posterior responde directamente la pregunta aplicada: ¿cuánto baja el techo de crecimiento alcanzable cuando la violencia rezagada pasa de su percentil 10 empírico al percentil 90?

| Cuantil | Techo, violencia baja (p10) | Techo, violencia alta (p90) | Caída | 94% HDI (caída) | P(caída>0 \| datos) |
|---|---|---|---|---|---|
| 0.90 | 8.79 pts | 5.52 pts | 3.27 pts | [1.09, 5.45] | 0.997 |
| 0.95 | 10.29 pts | 6.43 pts | 3.86 pts | [2.06, 5.61] | >0.999 (positiva en cada muestra posterior) |

Advertencias: esto sigue siendo exploratorio —motivado por un hallazgo de revisión rápida, no una hipótesis pre-registrada—. G=11 sigue siendo pequeño incluso para un modelo jerárquico; el partial pooling regulariza pero no puede generar información que los datos no contienen, y los priors son débilmente informativos, no planos. La verosimilitud Asimétrica de Laplace estima un cuantil a la vez y no garantiza por sí misma cuantiles monótonos en tau —de hecho, el hallazgo central es precisamente que el efecto NO es monótono: nulo en la mediana, negativo solo en la cola superior.

## Extensión del Índice Institucional (Septiembre 2026)

El índice institucional (`inst_avg` / `inst_pca`) antes usaba solo 3 de las 6 dimensiones de los Worldwide Governance Indicators (WGI) de Kaufmann et al. (2010) (estado de derecho, control de corrupción, estabilidad política) —un subconjunto arbitrario, no la construcción estándar. Ahora usa las seis, agregando voz y rendición de cuentas, efectividad gubernamental y calidad regulatoria. También se agregaron remesas (% del PIB), un canal económico de primer orden en la región (solo El Salvador ronda 20-25% del PIB), probadas como control de robustez.

**El índice de 6 dimensiones es internamente coherente**: alfa de Cronbach = 0.922 (más alto que con 3 dimensiones), KMO = 0.863, PC1 explica 72.9% de la varianza de las seis dimensiones, y todos los loadings del PCA son positivos (0.31–0.46) —un solo factor de "calidad de gobernanza general" subyace claramente en las seis series, e inst_avg/inst_pca siguen siendo casi idénticos (r = 0.9992).

**Pero la expansión cambia materialmente los resultados de EQ1 y EQ2, y el hallazgo honesto es que son menos robustos de lo que sugería el índice de 3 dimensiones:**

| | EQ1 (violencia→instituciones) | EQ2 (instituciones→IED) | EQ3 (IED→crecimiento) |
|---|---|---|---|
| p (clusterizado), 3→6 dim. | 0.093 → 0.462 | 0.030 → 0.030 | 0.010 → 0.010 |
| p (Driscoll-Kraay) | 0.010 → 0.310 | 0.006 → 0.006 | — |
| p (bootstrap de clúster silvestre) | 0.085 → 0.562 | 0.002 → 0.038 | 0.108 → 0.108 |
| ¿Sobrevive rezago de 1 año? | — | Sí (p=0.036) → **No (p=0.318)** | — |
| ¿Sobrevive tendencias por país? | — | Sí (p=0.077) → **No (p=0.260)** | — |

EQ1 (violencia → instituciones) ya no es significativo bajo ningún estimador. Probar cada dimensión WGI individualmente explica por qué: `rule_of_law` por sí sola sigue siendo significativa (coef=-1.96, p=0.037) y la mayoría de las dimensiones apuntan en la dirección esperada (negativa), pero `voice_accountability` se mueve en la dirección *contraria* (coef=+1.72, p=0.339) —diluyendo el promedio compuesto. EQ2 (instituciones → IED) sigue siendo significativa en su especificación primaria (contemporánea), pero **ya no sobrevive** los chequeos de rezago de un año ni de tendencias por país que superaba con 3 dimensiones —su aparente robustez era en parte un artefacto del índice más estrecho. EQ3 está esencialmente sin cambios (su propia significancia proviene de `fdi_percent_gdp`, no de `inst_avg`). El bootstrap de mediación de la cadena completa se debilita aún más (efecto indirecto -0.011, IC 95% [-0.113, 0.020], p=0.548, vs. -0.031/p=0.144 con 3 dimensiones).

Esto se reporta como una advertencia genuina e importante, no se suaviza: **EQ1 siempre fue el eslabón más débil e inestable de este pipeline (ver cada sección de robustez anterior), y una construcción más completa y estándar del índice institucional muestra que no sobrevive en absoluto** —el eslabón violencia→instituciones debe leerse ahora como una hipótesis de trabajo que motiva el resto de la cadena, no como un resultado establecido. EQ2 sigue siendo el eslabón más defendible del pipeline, pero con una base de evidencia más estrecha que la documentada previamente.

**Chequeo de robustez con remesas** (agregadas como control, no como especificación primaria, ya que son en sí mismas un candidato a mediador/collider entre la emigración inducida por violencia y el crecimiento): EQ1 no se ve afectada (p=0.179, sigue sin ser significativa). EQ2 se fortalece sustancialmente al incluir remesas (coef 0.611→0.967, p=0.030→0.0013) —posiblemente porque las remesas absorben varianza que de otro modo confunde la relación instituciones-IED, aunque esta variante individual tampoco debe sobre-interpretarse. EQ3 no cambia (p=0.025, era p=0.018).

## Limitaciones

- **Número pequeño de grupos (N=11, antes N=8)**: Todos los métodos de inferencia (incluso el bootstrap de grupo silvestre y los errores estándar de Driscoll-Kraay) dependen de propiedades asintóticas que pueden no ser válidas con tan pocos grupos independientes. Los intervalos de confianza deben interpretarse como sugestivos. La Extensión del Panel (arriba) muestra empíricamente cuánto pesaba esta restricción: subir de 8 a 11 países movió la significancia de EQ1/EQ2/EQ3 sustancialmente.
- **Endogeneidad potencial**: Aunque se utilizan tasas de homicidio rezagadas para abordar la simultaneidad inversa, las variables omitidas (por ejemplo, capacidad estatal, disturbance social) podrían afectar tanto la violencia como las instituciones.
- **Error de medición en las instituciones**: Los WGI son indicadores basados en percepciones y pueden contener error de medición.
- **Canal secuencial asumido**: El modelo asume un camino estricto violencia → instituciones → IED → crecimiento, pero podrían existir bucles de retroalimentación o efectos simultáneos. Un bootstrap formal del efecto indirecto conjunto (Revisión Metodológica, punto 2) confirma que esta cadena no supera un test de significancia como sistema con el tamaño muestral disponible.
- **Selección de variables de control**: Algunas variables de control tienen poca variación dentro de los países (por ejemplo, comercio como % del PIB), lo que limita lo que los efectos fijos pueden identificar. `gdp_per_capita_log` en particular es un control potencialmente post-tratamiento (Revisión Metodológica, punto 4).
- **No estacionariedad**: `homicide_rate_log`, `inst_avg` y `gdp_per_capita_log` no rechazan raíz unitaria en un test de panel Fisher-ADF, un riesgo de regresión espuria que los efectos fijos bidireccionales no corrigen por sí solos (Revisión Metodológica, punto 5).
- **Cointegración parcial (Módulo 07)**: dado que homicidios, instituciones y PIB per cápita son I(1), se probó formalmente si comparten una relación de equilibrio de largo plazo (Engle-Granger/Kao de dos pasos). Con el panel de 11 países, homicidios↔PIB per cápita (T3) sí muestra evidencia de cointegración (p=0.011) y un modelo de corrección de errores con velocidad de ajuste significativa (phi=-0.109, p<0.01). Violencia↔instituciones (T1) e instituciones↔PIB per cápita (T2) siguen sin mostrar evidencia de cointegración, lo que sigue ofreciendo una explicación estructural —más allá del poder estadístico— de por qué EQ1 es el eslabón más inestable de los tres en este pipeline.

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
│       ├── 07_cointegration.py
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