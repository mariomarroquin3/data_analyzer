# Hipótesis de Investigación
## Seguridad, Gobernanza y Desarrollo Económico en Centroamérica (2000-2024)

> Documento de trabajo.
>
> Estas hipótesis representan el marco teórico preliminar del proyecto y deberán ser reevaluadas una vez se obtengan resultados mediante modelos de panel, efectos fijos, regresiones múltiples y técnicas de Machine Learning.
>
> Ninguna hipótesis se considera confirmada hasta completar el análisis estadístico final.

---

# 1. Pregunta de investigación

## Pregunta principal

¿De qué manera la seguridad influye sobre el desarrollo económico de los países de Centroamérica?

---

# 2. Evolución de la hipótesis

La hipótesis inicial del proyecto asumía una relación directa:

```text
Seguridad
    ↓
Crecimiento económico
```

Sin embargo, el análisis exploratorio y la literatura económica sugieren una relación más compleja.

La seguridad parece actuar como una manifestación observable de fenómenos institucionales más profundos relacionados con la gobernanza.

Por ello, la hipótesis evolucionó hacia un modelo institucional.

---

# 3. Hipótesis principal

## H1 – Hipótesis institucional

La violencia no afecta el crecimiento económico únicamente de forma directa.

La violencia deteriora la calidad institucional, especialmente el Estado de Derecho (Rule of Law), reduciendo la capacidad de atraer inversión y limitando el crecimiento económico.

La cadena teórica propuesta es:

```text
Violencia
    ↓
Rule of Law
    ↓
Inversión Extranjera (FDI)
    ↓
Crecimiento Económico
```

---

# 4. Hipótesis secundarias

## H2 – Violencia y Estado de Derecho

Un incremento en la tasa de homicidios se asocia con un deterioro del Estado de Derecho.

### Justificación

La violencia refleja dificultades del Estado para:

- ejercer control territorial,
- proteger derechos de propiedad,
- garantizar cumplimiento de contratos,
- aplicar la ley de forma efectiva.

### Evidencia preliminar

Modelo exploratorio:

```text
rule_of_law ~ homicide_rate_log
```

Resultados preliminares:

```text
Coeficiente ≈ -2.45
p < 0.001
R² ≈ 0.93
```

Interpretación preliminar:

Mayores niveles de violencia se asocian con una reducción significativa del indicador Rule of Law.

### Estado actual

Pendiente de validación mediante modelos de efectos fijos.

---

## H3 – Estado de Derecho e Inversión Extranjera

Mejoras en Rule of Law incrementan la inversión extranjera directa (FDI).

### Justificación

Los inversionistas requieren:

- estabilidad jurídica,
- protección de contratos,
- previsibilidad regulatoria.

### Evidencia preliminar

Modelo exploratorio:

```text
FDI ~ RuleOfLaw
```

Resultados preliminares:

```text
Coeficiente positivo
Estadísticamente significativo
R² ≈ 0.30
```

Interpretación preliminar:

La calidad institucional parece contribuir a la atracción de inversión extranjera.

### Estado actual

Pendiente de validación mediante efectos fijos.

---

## H4 – Inversión Extranjera y Crecimiento Económico

La inversión extranjera directa contribuye positivamente al crecimiento económico.

### Justificación

La inversión extranjera:

- incrementa el capital disponible,
- facilita transferencia tecnológica,
- impulsa productividad.

### Evidencia preliminar

Modelo exploratorio:

```text
GDP Growth ~ FDI
```

Resultados preliminares:

```text
Coeficiente ≈ +5.49
Significativo
```

Interpretación preliminar:

La inversión extranjera parece asociarse positivamente con el crecimiento económico.

### Estado actual

Pendiente de validación mediante panel data.

---

## H5 – Gobernanza y Desarrollo Económico

Los indicadores de gobernanza explican mejor el crecimiento económico que la violencia por sí sola.

Variables consideradas:

```text
Rule of Law
Political Stability
Control of Corruption
```

### Justificación

La violencia podría ser un síntoma observable de problemas institucionales más profundos.

### Pregunta clave

¿La violencia mantiene poder explicativo cuando se controla por gobernanza?

Si la respuesta es negativa:

```text
Violencia = síntoma
Gobernanza = causa subyacente
```

---

## H6 – Seguridad e Inversión Extranjera

Una reducción en la violencia incrementa la inversión extranjera.

### Cadena teórica

```text
Menos violencia
      ↓
Menor riesgo percibido
      ↓
Mayor inversión
```

### Estado actual

Hipótesis parcialmente respaldada de forma indirecta a través de Rule of Law.

Pendiente de análisis directo.

---

## H7 – Seguridad y Turismo

La reducción de la violencia incrementa la llegada de turistas.

### Cadena teórica

```text
Menos violencia
      ↓
Mayor percepción de seguridad
      ↓
Más turismo
```

### Resultados preliminares

La evidencia inicial es más débil que en el caso de FDI.

```text
R² ≈ 0.09
```

### Estado actual

Canal potencialmente secundario.

---

# 5. Hipótesis de mediación

## H8 – Canal institucional

La violencia afecta el crecimiento económico indirectamente mediante el deterioro institucional.

Modelo conceptual:

```text
Violencia
    ↓
Rule of Law
    ↓
FDI
    ↓
Growth
```

Hipótesis central del proyecto.

---

## H9 – Canal turístico

La violencia afecta el crecimiento económico mediante cambios en la actividad turística.

Modelo conceptual:

```text
Violencia
    ↓
Turismo
    ↓
Growth
```

Hipótesis alternativa.

Actualmente parece menos robusta que H8.

---

# 6. Endogeneidad

## Problema identificado

La relación entre gobernanza y crecimiento económico podría ser bidireccional.

Ejemplo:

```text
Rule of Law
      ↓
Growth
```

pero también:

```text
Growth
      ↓
Capacidad estatal
      ↓
Rule of Law
```

Por tanto:

- instituciones pueden generar crecimiento,
- crecimiento puede fortalecer instituciones.

---

## Implicación

Los modelos utilizados no permiten afirmar causalidad estricta.

Los resultados deberán interpretarse como:

```text
Asociaciones consistentes
```

y no como:

```text
Relaciones causales definitivas
```

---

# 7. Estrategia econométrica prevista

## Etapa 1

Análisis descriptivo:

- Estadísticas descriptivas
- Correlaciones
- Matriz de correlación

---

## Etapa 2

Diagnóstico:

- Missing values
- Outliers
- Multicolinealidad
- VIF

---

## Etapa 3

Regresión múltiple (OLS)

Modelos exploratorios iniciales.

---

## Etapa 4

Panel Data

### Fixed Effects (principal)

Objetivo:

Controlar diferencias estructurales entre países.

Ejemplo:

```text
GDPGrowth_it =
β1 RuleOfLaw_it +
β2 FDI_it +
α_i +
γ_t +
ε_it
```

---

### Random Effects (comparación)

Aplicar Hausman Test para selección de modelo.

---

## Etapa 5

Machine Learning

Modelos complementarios:

- Random Forest Regressor

Objetivos:

- Feature Importance
- SHAP Values
- Robustez predictiva

---

# 8. Posibles conclusiones esperadas

## Escenario A

La violencia mantiene significancia incluso controlando gobernanza.

Interpretación:

```text
La seguridad posee un efecto económico propio.
```

---

## Escenario B

La violencia pierde significancia al controlar gobernanza.

Interpretación:

```text
La seguridad actúa principalmente como indicador de calidad institucional.
```

---

## Escenario C

Rule of Law emerge como variable dominante.

Interpretación:

```text
La gobernanza es el mecanismo fundamental detrás
de las diferencias observadas en desarrollo económico.
```

---

# 9. Conclusión provisional

La hipótesis actualmente más consistente con la teoría y los resultados preliminares es:

> La violencia afecta el desarrollo económico principalmente de forma indirecta, mediante el deterioro del Estado de Derecho y de la calidad institucional, reduciendo la inversión extranjera y limitando el crecimiento económico.

No obstante, esta afirmación deberá ser reevaluada tras la estimación de modelos de panel con efectos fijos y la validación mediante técnicas de Machine Learning.