# AGENTS.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

## Contexto

Notebook `tp-final-airbnb-rj.ipynb`. Problema de regresión: predecir el precio por noche de alojamientos de Airbnb en Río de Janeiro (dataset detallado de Inside Airbnb, 90 columnas). Las 12 secciones están escritas y el notebook corre de punta a punta sin errores ni warnings. Las secciones 1–4 son del compañero; el resto es del asistente.

## Formato de los archivos markdown

Sin quiebres de línea artificiales dentro del texto. Dejar que cada párrafo fluya como una sola línea; el word wrap del lector se encarga. Sí dejar líneas en blanco entre párrafos y entre ítems de lista. Regla: si el editor envolvería la línea en la pantalla, no insertar `\n` explícito.

## Reglas duras

- **No instalar paquetes.** Usar sólo lo declarado en `pyproject.toml`: numpy, pandas, scipy, scikit-learn, xgboost, optuna, matplotlib, seaborn.
- **Markdown mínimo**, sin títulos decorativos. Solo el texto que aporte.
- **Comentarios en código, siempre muy descriptivos.** Regla permanente, no sólo para lo ya escrito: todo código nuevo que se agregue (cualquier sección) lleva comentarios que expliquen los conceptos de ML paso a paso — no sólo qué hace el código, sino por qué y qué significa cada pieza — para que el usuario pueda leerlos sin experiencia previa en ML. Son "úsalos y tiralos": el usuario los borra él mismo a medida que lee cada parte, así que no hay que ahorrar detalle pensando en que van a quedar para siempre. Las secciones 5–11 ya están así (ver sesión donde se aplicó); mantener el mismo nivel de detalle en cualquier celda nueva.
- **Markdown de las secciones 5–12**: tono de observación, lacónico, sin explicar qué es cada algoritmo (el lector es un profesor, no alguien aprendiendo ML). Justificar decisiones específicas del dataset sí; explicar conceptos de libro no.
- **No sobrecomplicar.** Si el curso ya tiene una forma de hacer algo, copiarla.
- **No tocar las secciones 1 a 4** (son del compañero, incluida su celda de EDA/preprocesamiento), salvo para arreglar un error o warning existente — y ahí, lo mínimo indispensable.
- **Notebook debe correr sin warnings ni errores.** El bloque de imports tiene un único `warnings.filterwarnings("ignore")` que cubre todo (incluye un `UserWarning` informativo de `sklearn.utils.parallel` que aparece ~95 veces en el barrido de `max_features` de RF). Si se acota ese filtro en el futuro, correr el notebook completo después — puede destapar warnings que hoy están silenciados de más.
- **No entregar el cuaderno con el subsample de verificación activado.** La celda 6 tiene `df = df.sample(n=1000, random_state=SEMILLA).reset_index(drop=True)`. Para validar rápido: descomentada. **Para entregar: comentada.**
- **Referencias bibliográficas: no agregar.** El enunciado las pide (`trabajo_final/README.md`), pero es una decisión explícita del usuario. No agregar la sección sin que lo pida de nuevo.
- **Mantener este archivo actualizado** ante cualquier cambio de diseño o hallazgo nuevo — es la única fuente de contexto persistente del trabajo.

## Helpers ya disponibles en el notebook (no redefinir)

- `construir_preprocesador()` — `ColumnTransformer` con `SimpleImputer(median)` + `StandardScaler` para numéricas, `OneHotEncoder(drop='first')` para categóricas, `passthrough` para binarias.
- `metricas(y_real, y_pred, nombre)` — devuelve MAE, RMSE, MAPE, R² en BRL.
- `evaluar(nombre, modelo, log_target=True)` — entrena, evalúa y devuelve fila lista para apilar. Por defecto entrena sobre `log(precio)` y mide en BRL. Pasar `log_target=False` únicamente en el experimento crudo vs log de 3.7. **Muta el objeto `modelo` que recibe** (le hace `.fit(X_train, objetivo_train)` adentro) — después de `historial.append(evaluar("X", modelo))`, `modelo` ya queda entrenado; no volver a fitearlo para leer coeficientes o importancias.
- `resultados` — lista con los 2 baselines. `exp` — lista con los dos OLS de 3.7 (crudo y log). `historial` — tabla viva donde se apila cada modelo nuevo (`historial.append(evaluar("Nombre", pipeline))`); la sección 11 arma `df_final` a partir de acá.
- `XGB_FIJOS` — dict con los kwargs fijos de XGBoost (`objective="reg:absoluteerror", random_state=SEMILLA, n_jobs=-1`), usado con `**XGB_FIJOS` en 9.3, 9.4 y 10.1.
- Constantes y paleta (`AZUL`, `MAGENTA`, `VERDE`, `NARANJA`, `CELESTE`, `GRIS`, `BASELINE`, `SEMILLA=42`) ya definidas arriba.

## Convención de target y `evaluar()`

Por decisión de 3.7 todos los modelos nuevos entrenan sobre `log(precio)`. Por eso `evaluar(name, model)` (sin flag) ya hace lo correcto. No repetir `log_target=True` en cada llamada.

## Métrica principal

**MAE en BRL** (medido siempre en reales, no en log). RMSE/MAPE/R² como secundarias.

## Antes de entregar

1. Comentar la línea de subsample de la celda 6 y correr el notebook completo (estimado 4–9 horas; SVR, el barrido de RF y Stacking son los cuellos de botella). Si el runtime es prohibitivo, recortar en este orden: trials de Optuna → RF del sweep → RF del stacking → grilla de SVR.
2. Confirmar que termina sin errores ni warnings.
3. Releer las secciones 8, 9, 11 y 12 contra los resultados reales (ver hallazgos abajo) — varias afirmaciones se escribieron antes de tener una corrida completa.

## Hallazgos pendientes de re-verificación contra la corrida real

- **Importancia de barrio (secciones 8 y 12)**: afirman que las dummies de barrio están entre las features más importantes de RF. En una corrida con el subsample de verificación (n chico) esto puede ser **imposible de verificar**: si el umbral de 200 apariciones mínimas de 3.3 deja a `neighbourhood_cleansed` con una sola categoría, `OneHotEncoder(drop='first')` no genera ninguna columna para esa variable y no puede aparecer como importante. Confirmar que la afirmación se sostiene con la corrida real.
- En general, cualquier frase de las secciones 8–12 que explique *por qué* gana un modelo (no sólo *que* gana) conviene releerla después de la corrida real.
