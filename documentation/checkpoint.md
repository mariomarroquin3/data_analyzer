# CHECKPOINT – CADENA ECONOMÉTRICA (VIOLENCIA → INSTITUCIONES → INVERSIÓN → CRECIMIENTO)

## 1. ¿Tu teoría es correcta?

Sí, en términos económicos es totalmente consistente con la literatura:

- La violencia afecta la capacidad estatal y la seguridad jurídica  
- Eso debilita instituciones (rule of law, corrupción, estabilidad política)  
- Eso reduce incentivos a invertir (FDI, turismo)  
- Eso reduce acumulación de capital → menor crecimiento económico  

Esto es consistente con:
- Acemoglu & Robinson (instituciones y desarrollo)
- World Bank Governance Indicators literature
- Literatura de “violence traps” en desarrollo económico

---

## 2. ¿Qué parte de tus resultados la confirma?

### A → B (Violencia → Instituciones)

Resultado clave:
- homicide_rate_log → rule_of_law = -2.45 (p < 0.001)
- R² = 0.934

Interpretación:
- Evidencia muy fuerte
- Relación estable y altamente significativa
- Consistente con teoría institucional

👉 Este es el pilar más sólido del modelo

---

### B → C (Instituciones → inversión)

FDI:
- rule_of_law_hat → +0.55 (significativo)
- control_corruption → negativo significativo
- R² ≈ 0.30

Interpretación:
- Evidencia moderada-fuerte
- Dirección económica coherente
- Explicación parcial del mecanismo

Turismo:
- Efectos más débiles
- R² ≈ 0.09

⚠ Canal débil en esta muestra

---

### C → Growth (inversión → crecimiento)

Resultados:
- fdi_hat → positivo fuerte
- unemployment → negativo fuerte
- rule_of_law_hat → positivo fuerte
- R² ≈ 0.27

Interpretación:
- Coherente con macroeconomía estándar
- Crecimiento responde a inversión y calidad institucional

---

## 3. ¿Qué números prueban la teoría?

No es un solo coeficiente, sino una cadena:

### 🔗 Evidencia 1 (Violencia → instituciones)
- coeficiente: -2.45
- R²: 0.93

✔ Efecto estructural fuerte

---

### 🔗 Evidencia 2 (Instituciones → inversión)
- rule_of_law_hat → FDI: +0.55
- control corruption → -0.37
- R² ≈ 0.30

✔ Mecanismo parcialmente confirmado

---

### 🔗 Evidencia 3 (inversión → crecimiento)
- FDI_hat → growth: +5.49
- unemployment → growth: -0.67
- R² ≈ 0.27

✔ Canal macro consistente

---

## 4. Interpretación formal del modelo

El modelo sugiere:

> La violencia tiene un efecto indirecto negativo sobre el crecimiento económico a través del deterioro institucional y la reducción de inversión extranjera.

---

## 5. Limitaciones importantes

❌ No hay causalidad estricta (OLS encadenado ≠ causal inference)  
❌ No hay control explícito de efectos fijos en esta versión  
❌ Turismo es un canal débil en esta muestra  
