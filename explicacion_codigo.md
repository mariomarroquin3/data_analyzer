# Análisis del Código: Impacto de la Violencia en el Crecimiento Económico

---

## 1. Descripción de los Scripts Principales

### `eda.py` (Exploratory Data Analysis)
Realiza el análisis exploratorio inicial del dataset bruto. Identifica las columnas, cuenta las observaciones por país, la cobertura temporal y verifica la asimetría (skewness). Evalúa correlaciones preliminares (como la relación entre los homicidios y el crecimiento económico) y genera gráficos exploratorios (tendencias de homicidios por país, dispersiones y medias de indicadores institucionales).

### `01clean_panel.py` (Limpieza de Panel)
Toma el dataset inicial y lo prepara para modelos formales:
- **Estructura Panel:** Ordena por código de país y año.
- **Transformaciones Logarítmicas:** Aplica logaritmo natural al PIB per cápita, población y turismo (para reducir la varianza y permitir interpretaciones en términos de elasticidad).
- **Tratamiento de Outliers:** Aplica _winsorization_ al 1% y 99% a la tasa de homicidios (limita los valores extremos) para evitar que sesguen los modelos.
- **Salida:** Exporta los datos a `panel_ready_for_modeling.csv`.

### `02panel_interpolation.py` (Interpolación y PCA)
Toma los datos limpios y aborda los datos faltantes y la multicolinealidad:
- **Interpolación:** Aplica interpolación lineal a nivel de país para rellenar huecos, complementada con un relleno hacia adelante (ffill) y hacia atrás (bfill).
- **Reducción de Dimensionalidad (PCA):** Estandariza las variables y aplica Análisis de Componentes Principales para capturar el 90% de la varianza en componentes sintéticos. 
- **Salida:** `panel_pca_ready.csv`.

### `abc.py`
Prototipo básico usando `scikit-learn` de un modelo secuencial (en cadena) de 3 pasos:
1. **Modelo A:** La violencia predice la debilidad institucional (`rule_of_law`).
2. **Modelo B:** Las instituciones (predichas) determinan la Inversión Extranjera Directa (FDI) y el Turismo.
3. **Modelo C:** La FDI, el Turismo y las Instituciones determinan el crecimiento final (`gdp_growth`).

### `econometrics.py`
Es la versión formal econométrica de `abc.py`. En lugar de Machine Learning, utiliza `statsmodels` (OLS) para obtener estadísticos formales (R-cuadrado, valores p, errores estándar). Usa el mismo enfoque de "Ecuaciones Simultáneas" o Path Analysis, controlando por variables macroeconómicas. Genera `chain_model_final.csv`.

### `econometrics2.py`
Un análisis profundo y enfocado en un caso de estudio (El Salvador - SLV):
- **PCA Específico:** Crea un "Índice Institucional" reduciendo `rule_of_law`, `control_corruption` y `political_stability` a un solo componente.
- **Lags (Rezagos):** Utiliza variables rezagadas ($t-1$) para la violencia y la FDI, rompiendo la "endogeneidad" o causalidad bidireccional temporal.
- Genera visualizaciones y tablas normalizadas al estilo _paper_ académico.

### `randomforest.py`
Aplica aprendizaje automático no lineal (Random Forest Regressor) para predecir el crecimiento:
- Evalúa el poder predictivo real de los indicadores usando un _split_ temporal.
- Extrae la importancia general de las variables.
- Utiliza **SHAP values** para la explicabilidad local, identificando cómo la violencia o las instituciones impactan de forma no lineal (y visualiza las relaciones con gráficos _summary plot_).

---

## 2. Diferencias entre los Datasets Clave

1. **`final_research_dataset.csv`**: El dataset bruto consolidado original. Contiene variables macroeconómicas sin transformaciones logarítmicas, con datos faltantes (NaN) y outliers puros.
2. **`panel_ready_for_modeling.csv`**: El dataset panel "limpio". La tasa de homicidios está suavizada (winsorized) y muchas variables macro están en forma logarítmica. Ideal para correr modelos directos donde puedes eliminar filas con faltantes (dropna).
3. **`panel_pca_ready.csv`**: Versión imputada. No tiene valores faltantes gracias a la interpolación país por país, y añade Componentes Principales (PCA) para usar en modelos donde la colinealidad entre instituciones es un problema grave.
4. **`chain_model_final.csv`**: El dataset resultante de `econometrics.py`. Contiene las columnas originales más las **predicciones** de los modelos endógenos (e.g. `rule_of_law_hat`, `fdi_hat`). Sirve para auditar cómo se propagan los errores a través del modelo en cadena.

---

## 3. Base Matemática y Econométrica

El planteamiento metodológico detrás de `econometrics.py` y `econometrics2.py` descansa en varios pilares matemáticos:

### Análisis de Sendero (Path Analysis) / Mínimos Cuadrados en 2 Etapas Modificado
El código intenta resolver el problema de la "Caja Negra" económica, postulando que la violencia no destruye el PIB directamente, sino que destruye las instituciones, las cuales espantan la inversión, y esto reduce el PIB. Se expresa como un sistema recursivo de ecuaciones OLS:

1. **Ecuación Institucional (Efecto de la violencia):**
   $$ Inst_{i,t} = \beta_0 + \beta_1 \log(Hom_{i,t}) + \sum \phi X_{i,t} + \varepsilon_{1} $$
   Donde se asume $\beta_1 < 0$. Se extrae el valor ajustado ($\widehat{Inst_{i,t}}$).

2. **Ecuación de Inversión / Turismo (Canales de transmisión):**
   $$ \text{FDI}_{i,t} = \gamma_0 + \gamma_1 \widehat{Inst_{i,t}} + \sum \theta X_{i,t} + \varepsilon_{2} $$
   Se extrae el valor predicho ($\widehat{\text{FDI}_{i,t}}$). Usar valores ajustados en lugar de valores reales previene sesgos de atenuación.

3. **Ecuación de Crecimiento:**
   $$ \text{Crecimiento}_{i,t} = \alpha_0 + \alpha_1 \widehat{\text{FDI}_{i,t}} + \alpha_2 \widehat{Inst_{i,t}} + \varepsilon_{3} $$

### Rezagos Temporales (Lags) para Endogeneidad
En `econometrics2.py`, se introduce $Hom_{t-1}$. 
$$ Inst_{t} = \beta_0 + \beta_1 \log(Hom_{t-1}) + \varepsilon $$
**Justificación:** Existe causalidad inversa (la pobreza genera violencia, y la violencia pobreza). Al rezagar la variable (tomar el año anterior), nos aseguramos de que el choque de violencia ya ocurrió cuando se mide el impacto institucional de hoy. Esto reduce fuertemente el sesgo por simultaneidad.

### Análisis de Componentes Principales (PCA)
Las variables institucionales del Banco Mundial (Rule of Law, Corruption, Stability) están altamente correlacionadas (multicolinealidad), lo que infla los errores estándar (VIF altos). 
Matemáticamente, PCA busca un vector de pesos $w_{(1)}$ que maximice la varianza proyectada:
$$ \max_{||w|| = 1} w^T \Sigma w $$
En `econometrics2.py`, el primer componente (PCA1) se vuelve el `institution_index`, capturando la "salud institucional general" en una sola variable ortogonal, estabilizando la regresión OLS.

### Random Forest y Explicabilidad (Valores SHAP)
En `randomforest.py`, se rompe la asumpción de linealidad global OLS ($Y = X\beta + u$).
El bosque aleatorio promedia predicciones de árboles de decisión: $\hat{f} = \frac{1}{B} \sum_{b=1}^B f_b(x)$.
Para explicar este modelo "caja negra", el script utiliza **SHAP Values** basados en la Teoría de Juegos Coalicionales de Shapley. Matemáticamente, el valor SHAP ($\phi_j$) asigna el impacto marginal esperado de la característica $j$ promediado sobre todas las permutaciones posibles de características:
$$ \phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right] $$
Esto te permite concluir afirmaciones no lineales precisas, por ejemplo: _"Cuando la tasa de homicidios supera X límite, el impacto destructivo sobre el PIB se multiplica, restando $\phi_j$ puntos porcentuales al crecimiento"_.
