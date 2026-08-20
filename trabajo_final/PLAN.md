# PLAN.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

Estado actual del notebook: secciones 1 a 12 completas. Las secciones 5 a 9 corren tuning real (GridSearchCV para k-NN, SVR, árbol, RF; Optuna TPE para XGBoost); la sección 10 arma Voting y Stacking con las versiones limpias de las bases; la 11 arma la comparativa final con tabla y gráfico de barras; la 12 cierra con las conclusiones.

## Cambio menor ya aplicado en el notebook
Se flipeó el default de `evaluar(nombre, modelo, log_target=True)`. A partir de acá todas las llamadas a `evaluar(name, model)` entrenan sobre `log(precio)` sin necesidad de pasar el flag. Section 4 sigue pasándolo explícitamente (redundante pero inofensivo) y 3.7 sigue usándolo para comparar crudo vs log.

Convención: en cada sección 5–10, el modelo se entrena sobre `log(precio)` y la fila se apila con `historial.append(evaluar("Nombre", pipeline))`. Ojo: la tabla viva es `historial`, no `resultados` — en el notebook `resultados` quedó con los 2 baselines (celda de 3.6), `exp` guarda los dos OLS de 3.7, y la sección 4 creó `historial = resultados + exp` y apiló ahí Ridge y Lasso. La sección 11 arma `df_final` a partir de `historial`.

## 5. k-NN (clase 2)
- Importar `KNeighborsRegressor` (no está aún).
- `Pipeline([("prep", construir_preprocesador()), ("model", KNeighborsRegressor())])`.
- `GridSearchCV` sobre `model__n_neighbors` (`np.arange(1, 32, 2)`, impares de 1 a 31), `model__weights` (`["uniform","distance"]`), `model__p` (`[1, 2]`, Manhattan/Euclídea — las distancias con sentido pedagógico de la clase 2), `cv=5`, `scoring="neg_mean_absolute_error"`, `n_jobs=-1`. Grilla recortada a propósito: con 27k filas de train el costo de k-NN está en la predicción, y la grilla original (300 configs × 5 folds) tardaba horas; esta queda en 64 configs × 5 folds = 320 fits.
- Mostrar `best_params_` y `best_score_` (MAE de CV en log). Apilar `evaluar("k-NN", grid.best_estimator_)`.

## 6. SVR (clase 3)
- Importar `SVR`.
- `Pipeline([("prep", ...), ("model", SVR())])`.
- `GridSearchCV` con **grilla reducida**: `[{"model__C": [1, 10, 100], "model__kernel": ["linear"]}, {"model__C": [1, 10, 100], "model__kernel": ["rbf"], "model__gamma": ["scale"]}]`, `cv=5`, `scoring="neg_mean_absolute_error"`, `n_jobs=-1`. La grilla completa de `clase3/.../3 - SVM (Regresion).ipynb` (108 configs × 5 folds) está pensada para Hitters (~200 filas); con 27k filas de train cada fit RBF tarda minutos y la verificación final no terminaría nunca. Esta queda en 6 configs × 5 folds = 30 fits.
- Apilar `evaluar("SVR", grid.best_estimator_)`.

## 7. Árbol de regresión (clase 4)
- Importar `DecisionTreeRegressor`.
- Versión sin podar primero para mostrar sobreajuste (MAE train ≈ 0).
- Poda por `ccp_alpha`: usamos `cost_complexity_pruning_path` para tomar los alphas que sklearn ya prueba al entrenar un árbol sin restricción (no hace falta inventar la grilla). Si el path devuelve más de 50 alphas, sampleamos 50 equiespaciados dentro del rango para mantener el barrido manejable. El barrido mide MAE por CV 5-fold y grafica MAE vs número de hojas (estilo `clase4/.../1 - Árboles de regresión.ipynb`).
- **Desviación del plan original**: el plan proponía `np.linspace(0, 0.5, 200)` a mano. La versión con `cost_complexity_pruning_path` + submuestreo a 50 da una señal equivalente y se ejecuta ~4× más rápido, porque reutiliza los alphas que sklearn ya calculó.
- Apilar `evaluar("Árbol podado", best_tree)`.

## 8. Bagging y Random Forest (clase 5 nb 2)
- Importar `BaggingRegressor`, `RandomForestRegressor`, `permutation_importance`.
- Bagging: 200 árboles sobre remuestreo bootstrap, `n_jobs=-1`. Apilar `evaluar("Bagging", bagging)`.
- Random Forest: barrido `max_features ∈ np.arange(0.1, 1.01, 0.05)` con 200 árboles, registrar MAE de CV. Gráfico y elección del óptimo igual al de la clase. 19 valores × 5 folds = 95 fits.
- RF final con el mejor `max_features`, `n_estimators=500`, `oob_score=True`. Apilar `evaluar("Random Forest", rf_final)`.
- Cerrar con importancia por impureza y por permutación, dos `barh` side-by-side como en la clase.

## 9. Boosting y XGBoost (clase 5 nb 3)
- Importar `AdaBoostRegressor`, `GradientBoostingRegressor`, `xgb`, `optuna`.
- AdaBoost con la configuración razonable de la clase (árboles `max_depth=4`, `n_estimators=300`, `learning_rate=0.5`). Apilar `evaluar("AdaBoost", ada_bueno)`.
- Gradient Boosting (`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`). Mostrar la curva `staged_predict()` para evidenciar sobreajuste. Apilar `evaluar("Gradient Boosting", gb)`.
- XGBoost: tuneo por Optuna (TPE) sobre `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`. **`objective="reg:absoluteerror"`** porque la métrica es MAE. **`n_trials=100`** con CV 3-fold interno (300 fits). Apilar `evaluar("XGBoost (Optuna)", xgb_opt)`.
- **Desviación del plan original**: el plan proponía `n_trials=200`. 100 trials con TPE converge bien en 8 dimensiones y baja el runtime a la mitad. Si la corrida ya terminó y querés más señal, subir a 200 en `n_trials=`.

## 10. Voting y Stacking (clase 5 nb 1, versión regresión)
- Importar `VotingRegressor` y `StackingRegressor` (`Ridge` ya está importada en el bloque de imports).
- Reconstruir versiones "limpias" de las bases con los hiperparámetros ganadores (patrón de `clase5/.../4 - Ensambles...`: pipeline con `construir_preprocesador()` + modelo con params fijos, sin GridSearch interno). Bases: k-NN, árbol podado, Random Forest y XGBoost. **SVR queda afuera**: es la base más lenta y los ensambles la re-entrenan muchas veces — `VotingRegressor`/`StackingRegressor` clonan y re-ajustan las bases en cada fit, y encima `evaluar` hace 5-fold CV del ensamble completo (`StackingRegressor(cv=5)` multiplica los refits ~25× por base). Con SVR adentro serían horas.
- **Desviación del plan original**: el RF del stacking usa `n_estimators=200` en lugar de los 500 del modelo final de 8.3. El StackingRegressor clona cada base y la re-entrena 5 veces (cv interno) por cada fold del CV externo de `evaluar` — con 500 árboles la corrida se vuelve prohibitiva. 200 árboles mantiene la señal y reduce el costo unas 2.5×. El RF final de la sección 8 sigue siendo de 500 árboles.
- Como regresión no tiene `predict_proba`, se promedian predicciones — equivalente al soft voting de clasificación.
- `voting_reg = VotingRegressor(estimators=[("knn", knn), ("tree", tree), ("rf", rf), ("xgb", xgb)], n_jobs=-1)`. Apilar `evaluar("Voting", voting_reg)`.
- `StackingRegressor(estimators=mismos, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)`. Apilar `evaluar("Stacking", stacking_reg)`.
- Mostrar los coeficientes del meta-modelo (`final_estimator_.coef_`) en un `barh` corto, como en clase 5 nb 1.

## 11. Comparativa final del curso
- `df_final = pd.DataFrame(historial).set_index("modelo")`, redondear a 3 decimales, mostrar.
- Gráfico de barras con MAE de test ordenado. Paleta y estilo igual a la sección final de `clase5/.../3 - Boosting y XGBoost.ipynb`: baselines en gris, lineales en celeste, árbol en naranja, ensembles en gama magenta→rosa→violeta. Líneas horizontales de referencia con el baseline heurístico y el mejor modelo previo (Ridge o Lasso).
- Markdown breve, sin números hardcoded: describe qué familia gana, qué pasa entre lineales y ensambles, si voting/stacking agregan valor por encima de los homogéneos. Los valores exactos están en la tabla de arriba y se leen de ahí.

## 12. Conclusiones
- Markdown breve, mismo tono que el resto del trabajo. **Sin números hardcoded**: el ganador se lee del `df_final` de la sección 11; el resto son patrones cualitativos (qué familia dominó, qué features pesaron, limitaciones honestas).
- Limitaciones honestas: target recortado en 2.4, dataset cross-sectional sin estacionalidad ni eventos, reseñas faltantes imputadas a cero, y grillas recortadas por runtime (k-NN: 64 configs en vez de 300, SVR: 6 configs en vez de 108, XGBoost Optuna: 100 trials en vez de 200, stacking RF: 200 árboles en vez de 500).

## Referencias bibliográficas
- **No agregar.** El enunciado las pide, pero el usuario decidió dejarlas fuera del alcance del notebook. Si una sesión futura ve esto, no crear la sección.

## Verificación final
- Convertir el notebook ejecutándolo: `uv run jupyter nbconvert --execute --to notebook trabajo_final/tp-final-airbnb-rj.ipynb` (o abrirlo en Jupyter y correr todo). **Esperado**: tarda entre 4 y 9 horas en una laptop de 4–8 cores (SVR, el barrido de Random Forest y Stacking son los cuellos de botella). Si el runtime es prohibitivo, los recortes documentados arriba son los primeros candidatos a achicar (en orden: trials de Optuna, RF del sweep, RF del stacking, grilla de SVR).
- Comprobar que termina sin errores ni warnings. Si aparece alguno nuevo, apagarlo en el bloque de imports con comentario explicando por qué.

## Verificación previa (smoke test)
- Antes de la corrida real, conviene una pasada de validación que confirme que el código cierra de punta a punta sin errores ni warnings. La celda 6 del cuaderno (justo después de la carga del dataset) tiene un toggle comentado para subsamplear `df`. **Para entrega: dejar comentada.** Para validación: descomentar y ejecutar.
- Flujo recomendado:
  1. Abrir el cuaderno, ir a la celda 6 y descomentar la línea `# df = df.sample(n=1500, random_state=SEMILLA).reset_index(drop=True)`. Opcionalmente cambiar `n=1500` por un valor mayor si querés un smoke más fiel.
  2. Ejecutar todas las celdas (Kernel → Restart & Run All en Jupyter, o `uv run jupyter nbconvert --execute --to notebook --inplace trabajo_final/tp-final-airbnb-rj.ipynb` desde la terminal).
  3. Inspeccionar: cada celda de código debe tener outputs sin `error` ni `stream` con `Warning`/`FutureWarning`/`UserWarning`/`RuntimeWarning`. Las tablas en 3.6, 4.1, 5.1, 6.1, 7.3, 8.1, 8.3, 9.1, 9.2, 9.4, 10.2, 10.3, 11.1 deben renderizar con números finitos.
  4. Si algo se rompe, arreglar in situ y volver a ejecutar.
  5. Una vez verde, **volver a comentar la línea de subsample** antes de la corrida real.
- **No entregar la versión smoke**: las métricas son humo (train y test son chicos), sólo sirven para validar que el código corre.

## Archivos generados en este turno
- `trabajo_final/AGENTS.md`
- `trabajo_final/PLAN.md`
- Patch mínimo en `tp-final-airbnb-rj.ipynb`: default de `evaluar(..., log_target=)` cambiado de `False` a `True`.

## Archivos modificados en este turno (revisión del notebook)
- `tp-final-airbnb-rj.ipynb`: secciones 5–10 reemplazadas con código real (GridSearchCV, Optuna, RF sweep, Voting, Stacking). Narrativas de las secciones 5, 6, 7, 8, 9, 11 y 12 reescritas sin números hardcoded — los valores que se lean ahora van a salir del código. Las secciones 1–4 están intactas salvo el cambio de default de `evaluar` documentado arriba.

## Archivos modificados en este turno (wiring del smoke test)
- `tp-final-airbnb-rj.ipynb`:
  - **Bug pre-existente arreglado en celda 8.4** (permutation importance): `nombres = [c.split("__", 1)[1] for c in rf.named_steps["prep"].get_feature_names_out()]` producía 61 nombres (post-OneHot) pero `permutation_importance(rf, X_test, ...)` devuelve 32 importances (raw columns, antes del preprocesador). Ahora hay dos índices separados (`nombres_post` para `feature_importances_` y `nombres_raw` para permutation); el barh de impureza usa `orden_impureza` y el de permutación usa `orden`. Se quitó `sharey=True` porque los dos paneles ya no comparten eje. **Sin esta fix el cuaderno revienta en la sección 8.4 incluso en la corrida real.**
  - **Bug pre-existente arreglado en celda 8.2** (RF sweep): `np.arange(0.1, 1.01, 0.05)` produce `1.0000000000000004` por error de punto flotante, y sklearn >=1.5 rechaza valores `> 1.0` en `max_features`. Cambió a `np.minimum(np.arange(0.1, 1.01, 0.05), 1.0)` con comentario corto.
  - **Warnings filtrados en bloque de imports**: agregado `warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")` con comentario. Es un `UserWarning` informativo que sklearn >=1.5 emite cada vez que `cross_val_predict` se usa con `n_jobs=-1` fuera de un `Parallel(...)` propio — ruido interno, no afecta correctness, pero se imprime ~95 veces en el barrido de max_features.
  - Nueva celda 6 (después de la carga del dataset) con un toggle de subsample comentado: `# df = df.sample(n=1500, random_state=SEMILLA).reset_index(drop=True)`. Descomentar la línea para correr todo el cuaderno sobre una muestra chica y verificar que cierra sin errores ni warnings antes de la corrida larga. Para la entrega, dejar comentadas.

## Cómo correr la verificación rápida
- Abrir el cuaderno, ir a la celda 6 (justo después de la carga del dataset) y descomentar la línea de subsample. Cambiar `n=1500` por un número mayor (5000–10000) si querés un smoke más fiel pero todavía rápido.
- Ejecutar todas las celdas (Kernel → Restart & Run All en Jupyter, o `uv run jupyter nbconvert --execute --to notebook --inplace trabajo_final/tp-final-airbnb-rj.ipynb` desde la terminal).
- El smoke subsamplea `df` antes del split: cambia el tamaño de train **y** de test. Por eso las métricas en las tablas son sólo para validar — la entrega debe correr con la línea comentada.
- Esperado: el cuaderno cierra en minutos con todos los modelos entrenados, tablas y gráficos renderizados, cero errores y cero warnings.
