# AGENTS.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

## Contexto
Notebook `tp-final-airbnb-rj.ipynb`. Problema de regresión: predecir el precio por noche de alojamientos de Airbnb en Río de Janeiro (dataset detallado de Inside Airbnb, 90 columnas). Faltan las secciones 5 a 12 (modelos no lineales, voting/stacking, comparativa final y conclusiones). El trabajo previo del compañero (secciones 1–4) está preservado tal cual.

## Materiales fuente
Archivos ya leídos durante la sesión inicial. Reabrir directamente si hace falta; no hace falta buscarlos de nuevo.

### Documentos del curso y del examen
- `README.md` (raíz del repo) — descripción del curso y temario por clase.
- `trabajo_final/README.md` — enunciado del trabajo práctico final (estructura sugerida: introducción, datos, entrenamiento, métricas, conclusiones).
- `pyproject.toml` — dependencias del proyecto (lo único permitido instalar).

### Trabajos previos del examen
- `trabajo_final/tp-previo-airbnb-ba.ipynb` — antecedente del compañero sobre Buenos Aires. Predice reseñas porque el precio venía 100 % nulo; sirve como referencia de estructura y preprocesamiento.
- `trabajo_final/tp-final-airbnb-rj.ipynb` — el notebook actual a completar.

### Notebooks clase 2 (`clase2/jupyter_notebooks/`)
Base para k-NN, frameworks de búsqueda y regresión logística.
- `1 - Medidas de distancia.ipynb` — distancia Minkowski, base conceptual de k-NN.
- `2 - Clasificacion.ipynb` — ejemplo completo de clasificación con baseline, F1, matriz de confusión y comparativa final.
- `3 - Hiper-parametros.ipynb` — `GridSearchCV` y `RandomizedSearchCV`.
- `4 - Optimizacion bayesiana.ipynb` — concepto de optimización bayesiana.
- `5 - Framework Optuna.ipynb` — patrón `objective` + `TPESampler` (reusado en XGBoost).
- `Auxiliar - Regresion Logistica.ipynb` — regresión logística con `Pipeline`.
- `EDA - Pima Diabetes.ipynb` — EDA del dataset Pima.

### Notebooks clase 3 (`clase3/jupyter_notebooks/`)
Base para SVR y modelos lineales regularizados.
- `1 - SVM (Clasificacion).ipynb` — SVC, mezclada con k-NN y logística.
- `2 - Caso práctico.ipynb` — caso de aplicación.
- `3 - SVM (Regresion).ipynb` — patrón de `GridSearchCV` con tres listas de kernels (linear / rbf+sigmoid / poly); **referencia directa para sección 6**.
- `Auxiliar - Regresion Lineal.ipynb` — OLS, Ridge, Lasso, multicolinealidad.
- `EDA - Hitters.ipynb` — EDA del dataset Hitters, base del baseline por tramos.

### Notebooks clase 4 (`clase4/jupyter_notebooks/`)
Base para árboles de regresión.
- `1 - Árboles de regresión.ipynb` — poda con `ccp_alpha` y barrido; **referencia directa para sección 7**.
- `2 - Árboles de clasificación.ipynb` — búsqueda Optuna para `DecisionTreeClassifier`.

### Notebooks clase 5 (`clase5/jupyter_notebooks/`)
Las más usadas, base para ensambles y la comparativa final.
- `1 - Modelos que votan.ipynb` — `VotingClassifier`, `StackingClassifier`, matriz de correlación de errores, `buscar_umbral`; **referencia directa para sección 10 (versión `Regressor`)**.
- `2 - Bagging y Random Forest.ipynb` — `BaggingRegressor`, `RandomForestRegressor`, barrido de `max_features`, `permutation_importance`; **referencia directa para sección 8**.
- `3 - Boosting y XGBoost.ipynb` — `AdaBoostRegressor`, `GradientBoostingRegressor` con `staged_predict`, XGBoost tuneado con Optuna; **referencia directa para sección 9**.
- `4 - Ensambles en clasificación y comparativa final.ipynb` — **ESTILO OBJETIVO** de la sección 11 de comparativa final. Replicar su estructura de "Comparativa final del curso": tabla con todos los modelos + gráfico de barras + observaciones. Trabaja sobre Pima (clasificación), pero la **estructura visual y narrativa** es la que debemos imitar; nosotros intercambiamos F1 por MAE en BRL y el target predicho por `log(precio)`.

## Formato de los archivos markdown
Sin quiebres de línea artificiales dentro del texto. Dejar que cada párrafo fluya como una sola línea; el word wrap del lector se encarga. Sí dejar líneas en blanco entre párrafos y entre ítems de lista. Regla: si el editor envolvería la línea en la pantalla, no insertar `\n` explícito.

## Reglas duras
- **No instalar paquetes.** Usar sólo lo declarado en `pyproject.toml`: numpy, pandas, scipy, scikit-learn, xgboost, optuna, matplotlib, seaborn.
- **Markdown mínimo**, sin títulos decorativos. Solo el texto que aporte.
- **Comentarios excesivos en código** (luego los borro yo): explicar qué hace cada bloque, por qué se hace, y qué devuelve.
- **No sobrecomplicar.** Reproducir el patrón de los notebooks de las clases 2–5; si el curso ya tiene una forma de hacer algo, copiarla.
- **No rehacer lo que ya está escrito** (secciones 1 a 4), salvo para arreglar un error o warning existente: en ese caso se corrige in situ lo mínimo indispensable. En todos los demás casos, sí se pueden agregar secciones nuevas debajo.
- **Notebook debe correr sin warnings ni errores.** Los `FutureWarning` se silencian ya con `warnings.filterwarnings("ignore")` en el bloque de imports; si aparece uno nuevo al sumar código, agregar la línea de supresión en ese mismo bloque con un comentario corto.
- **Mantener ambos archivos (`AGENTS.md` y `PLAN.md`) actualizados** ante cualquier cambio de diseño (ruta nueva, regla nueva, sección nueva) para que una sesión futura no pierda contexto.

## Helpers ya disponibles en el notebook (no redefinir)
- `construir_preprocesador()` — `ColumnTransformer` con `SimpleImputer(median)` + `StandardScaler` para numéricas, `OneHotEncoder(drop='first')` para categóricas, `passthrough` para binarias.
- `metricas(y_real, y_pred, nombre)` — devuelve MAE, RMSE, MAPE, R² en BRL.
- `evaluar(nombre, modelo, log_target=True)` — entrena, evalúa y devuelve fila lista para apilar. Por defecto entrena sobre `log(precio)` y mide en BRL. Pasar `log_target=False` únicamente en el experimento crudo vs log de 3.7. Es una adaptación de la `evaluar()` de clase 5 nb 2: usa `cross_val_predict` y destransforma con `np.exp` para medir el MAE de CV en BRL (la de la clase usa `cross_val_score` en la escala cruda del target). No redefinirla.
- `resultados` — lista con los 2 baselines (creada en 3.5, cargada en 3.6). No recibe más filas.
- `exp` — lista con los dos OLS del experimento 3.7 (crudo y log).
- `historial` — **tabla viva donde se apila cada modelo nuevo** (nace en 4.1 como `resultados + exp`, ya tiene Ridge y Lasso). Las secciones 5–10 apilan acá con `historial.append(evaluar("Nombre", pipeline))` y la sección 11 arma el `df_final` a partir de `historial`.
- Constantes y paleta (`AZUL`, `MAGENTA`, `VERDE`, `NARANJA`, `CELESTE`, `GRIS`, `BASELINE`, `SEMILLA=42`) ya definidas arriba.

## Patrones a copiar tal cual de las clases
- Tuning por grilla (`GridSearchCV`) para modelos con pocos hiperparámetros (k-NN, SVR, árbol, RF). Ver `clase3/.../3 - SVM (Regresion).ipynb`. **Excepción de runtime**: las grillas de clase asumen datasets chicos (Hitters tiene ~200 filas; acá train son ~27k). k-NN y SVR usan grillas reducidas — ver secciones 5 y 6 del PLAN.
- Tuning por Optuna sólo para XGBoost. Ver `clase5/.../3 - Boosting y XGBoost.ipynb` (función `objective` + `champion_callback`).
- Estilo del gráfico de comparativa final: ver la sección final de `clase5/.../3 - Boosting y XGBoost.ipynb` y de `clase5/.../2 - Bagging y Random Forest.ipynb`.
- `permutation_importance` para interpretar el modelo ganador (clase 5 nb 2).
- `staged_predict()` para visualizar sobreajuste en `GradientBoostingRegressor` (clase 5 nb 3).
- `VotingRegressor` y `StackingRegressor` para la sección 10, calcados del patrón de `clase5/.../1 - Modelos que votan.ipynb` (con `Regressor` en lugar de `Classifier`). El meta-learner de stacking será una `Ridge` simple. Bases: k-NN, árbol podado, Random Forest y XGBoost — **SVR queda afuera** porque los ensambles clonan y re-entrenan las bases en cada fit (y el CV de `evaluar` lo multiplica ~25× en stacking).

## Convención de target y `evaluar()`
Por decisión de 3.7 todos los modelos nuevos entrenan sobre `log(precio)`. Por eso `evaluar(name, model)` (sin flag) ya hace lo correcto. No repetir `log_target=True` en cada llamada.

## Métrica principal
**MAE en BRL** (medido siempre en reales, no en log). RMSE/MAPE/R² como secundarias.

## Salida esperada
Estructura del trabajo terminado, en orden:
1. Secciones 1 a 4 que ya están hechas (intacto).
2. Secciones 5 a 9: una por modelo nuevo (k-NN, SVR, árbol podado, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost), cada una con su fila apilada a `historial` vía `evaluar()`.
3. **Sección 10 — Voting y Stacking**: `VotingRegressor` y `StackingRegressor` combinando versiones limpias de k-NN, árbol podado, Random Forest y XGBoost (sin SVR; ver sección 10 del PLAN).
4. **Sección 11 — Comparativa final del curso** (la pieza central que el usuario pidió priorizar):
- Misma estructura que la sección homónima de `clase5/.../4 - Ensambles en clasificación y comparativa final.ipynb`, adaptada a regresión:
- Tabla con todos los modelos (baselines + OLS/Ridge/Lasso + 5–9 + Voting + Stacking), todas las métricas (`MAE BRL`, `RMSE BRL`, `MAPE`, `R²`, y `MAE CV` cuando aplique).
- Gráfico de barras con el MAE de test ordenado, paleta y formato igual a la sección final de `clase5/.../3 - Boosting y XGBoost.ipynb` (baselines en gris, lineales en celeste, árbol en naranja, ensembles en gama magenta→rosa→violeta, líneas de referencia horizontales para el baseline heurístico y el mejor modelo previo).
- Observaciones en markdown (corto), mismo tono que el curso: qué familia gana, qué pasa entre lineales y ensambles, si voting/stacking agregan valor por encima de los homogéneos.
5. **Sección 12 — Conclusiones**: cierre breve (modelo ganador, mejora vs baselines y vs el techo lineal, features dominantes, limitaciones). **Referencias bibliográficas: no agregar** — el enunciado las pide pero el usuario decidió dejarlas fuera del notebook.
