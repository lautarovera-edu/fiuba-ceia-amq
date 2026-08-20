# PLAN.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

Estado actual del notebook: secciones 1 (intro) a 4 completas (incluye 4.3, observaciones de los modelos lineales). Faltan las secciones 5 a 12. La numeración continúa la del compañero.

## Cambio menor ya aplicado en el notebook
Se flipeó el default de `evaluar(nombre, modelo, log_target=True)`. A partir de acá todas las llamadas a `evaluar(name, model)` entrenan sobre `log(precio)` sin necesidad de pasar el flag. Section 4 sigue pasándolo explícitamente (redundante pero inofensivo) y 3.7 sigue usándolo para comparar crudo vs log.

Convención: en cada sección 5–10, el modelo se entrena sobre `log(precio)` y la fila se apila con `historial.append(evaluar("Nombre", pipeline))`. Ojo: la tabla viva es `historial`, no `resultados` — en el notebook `resultados` quedó con los 2 baselines (celda de 3.6), `exp` guarda los dos OLS de 3.7, y la sección 4 creó `historial = resultados + exp` y apiló ahí Ridge y Lasso. La sección 11 arma `df_final` a partir de `historial`.

## 5. k-NN (clase 2)
- Importar `KNeighborsRegressor` (no está aún).
- `Pipeline([("prep", construir_preprocesador()), ("model", KNeighborsRegressor())])`.
- `GridSearchCV` sobre `model__n_neighbors` (`np.arange(1, 32, 2)`, impares de 1 a 31), `model__weights` (`["uniform","distance"]`), `model__p` (`[1, 2]`, Manhattan/Euclídea — las distancias con sentido pedagógico de la clase 2), `cv=5`, `scoring="neg_mean_absolute_error"`, `n_jobs=-1`. Grilla recortada a propósito: con 27k filas de train el costo de k-NN está en la predicción, y la grilla original (300 configs × 5 folds) tardaba horas; esta queda en 64 configs.
- Mostrar `best_params_`. Apilar `evaluar("k-NN", grid.best_estimator_)`.

## 6. SVR (clase 3)
- Importar `SVR`.
- `Pipeline([("prep", ...), ("model", SVR())])`.
- `GridSearchCV` con **grilla reducida**: `[{"model__C": [1, 10, 100], "model__kernel": ["linear"]}, {"model__C": [1, 10, 100], "model__kernel": ["rbf"], "model__gamma": ["scale"]}]`, `cv=5`, `scoring="neg_mean_absolute_error"`, `n_jobs=-1`. La grilla completa de `clase3/.../3 - SVM (Regresion).ipynb` (108 configs × 5 folds) está pensada para Hitters (~200 filas); con 27k filas de train cada fit RBF tarda minutos y la verificación final no terminaría nunca. Esta queda en 6 configs × 5 folds = 30 fits.
- Apilar `evaluar("SVR", grid.best_estimator_)`.

## 7. Árbol de regresión (clase 4)
- Importar `DecisionTreeRegressor`.
- Versión sin podar primero para mostrar sobreajuste (MAE train ≈ 0).
- Poda por `ccp_alpha`: barrido `np.linspace(0, 0.5, 200)` midiendo MAE por CV 5-fold; gráfico MAE vs nº de hojas (estilo `clase4/.../1 - Árboles de regresión.ipynb`). Mismo rango que la clase, que también entrena el árbol sobre target en log (`np.log(Salary)`); un rango más angosto (0–0.05) podía cortar antes del óptimo.
- Apilar `evaluar("Árbol podado", best_tree)`.

## 8. Bagging y Random Forest (clase 5 nb 2)
- Importar `BaggingRegressor`, `RandomForestRegressor`, `permutation_importance`.
- Bagging: 200 árboles sobre remuestreo bootstrap, `n_jobs=-1`. Apilar `evaluar("Bagging", bagging)`.
- Random Forest: barrido `max_features ∈ np.arange(0.1, 1.01, 0.05)` con 200 árboles, registrar MAE de CV. Gráfico y elección del óptimo igual al de la clase.
- RF final con el mejor `max_features`, `n_estimators=500`, `oob_score=True`. Apilar `evaluar("Random Forest", rf_final)`.
- Cerrar con importancia por impureza y por permutación, dos `barh` side-by-side como en la clase.

## 9. Boosting y XGBoost (clase 5 nb 3)
- Importar `AdaBoostRegressor`, `GradientBoostingRegressor`, `xgb`.
- AdaBoost con la configuración razonable de la clase (árboles `max_depth=4`, `n_estimators=300`, `learning_rate=0.5`). Apilar `evaluar("AdaBoost", ada_bueno)`.
- Gradient Boosting (`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`). Mostrar la curva `staged_predict()` para evidenciar sobreajuste. Apilar `evaluar("Gradient Boosting", gb)`.
- XGBoost: tuneo por Optuna (TPE, `n_trials=200`) sobre `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`. Misma `objective` y `champion_callback` que la clase. Apilar `evaluar("XGBoost (Optuna)", xgb_opt)`.

## 10. Voting y Stacking (clase 5 nb 1, versión regresión)
- Importar `VotingRegressor` y `StackingRegressor` (`Ridge` ya está importada en el bloque de imports).
- Reconstruir versiones "limpias" de las bases con los hiperparámetros ganadores (patrón de `clase5/.../4 - Ensambles...`: pipeline con `construir_preprocesador()` + modelo con params fijos, sin GridSearch interno). Bases: k-NN, árbol podado, Random Forest y XGBoost. **SVR queda afuera**: es la base más lenta y los ensambles la re-entrenan muchas veces — `VotingRegressor`/`StackingRegressor` clonan y re-ajustan las bases en cada fit, y encima `evaluar` hace 5-fold CV del ensamble completo (`StackingRegressor(cv=5)` multiplica los refits ~25× por base). Con SVR adentro serían horas.
- Como regresión no tiene `predict_proba`, se promedian predicciones — equivalente al soft voting de clasificación.
- `voting_reg = VotingRegressor(estimators=[("knn", knn), ("tree", tree), ("rf", rf), ("xgb", xgb)], n_jobs=-1)`. Apilar `evaluar("Voting", voting_reg)`.
- Opcional: probar `weights` razonables bajando al k-NN (el más débil individualmente, suele ser el más ruidoso en el promedio).
- `StackingRegressor(estimators=mismos, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)`. Apilar `evaluar("Stacking", stacking_reg)`.
- Mostrar los coeficientes del meta-modelo (`final_estimator_.coef_`) en un `barh` corto, como en clase 5 nb 1.

## 11. Comparativa final del curso
- `df_final = pd.DataFrame(historial).set_index("modelo")`, redondear a 3 decimales, mostrar.
- Gráfico de barras con MAE de test ordenado. Paleta y estilo igual a la sección final de `clase5/.../3 - Boosting y XGBoost.ipynb`: baselines en gris, lineales en celeste, árbol en naranja, ensembles en gama magenta→rosa→violeta. Líneas horizontales de referencia con el baseline heurístico y el mejor modelo previo (Ridge o Lasso).
- Markdown breve: qué familia gana, qué pasa entre lineales y ensambles, si voting/stacking agregan valor por encima de los homogéneos.

## 12. Conclusiones
- Markdown breve, mismo tono que el resto del trabajo: qué modelo ganó y por cuánto sobre el baseline heurístico (387 BRL) y sobre el techo lineal (~289 BRL); qué features dominaron (tamaño, geografía, `sin_reseñas`); limitaciones honestas (precio cross-sectional sin temporada ni eventos, target recortado en 2.4, reseñas faltantes imputadas a cero).

## Referencias bibliográficas
- **No agregar.** El enunciado las pide, pero el usuario decidió dejarlas fuera del alcance del notebook. Si una sesión futura ve esto, no crear la sección.

## Verificación final
- Convertir el notebook ejecutándolo: `uv run jupyter nbconvert --execute --to notebook trabajo_final/tp-final-airbnb-rj.ipynb` (o abrirlo en Jupyter y correr todo).
- Comprobar que termina sin errores ni warnings. Si aparece alguno nuevo, apagarlo en el bloque de imports con comentario explicando por qué.

## Archivos generados en este turno
- `trabajo_final/AGENTS.md`
- `trabajo_final/PLAN.md`
- Patch mínimo en `tp-final-airbnb-rj.ipynb`: default de `evaluar(..., log_target=)` cambiado de `False` a `True`.
