# AGENTS.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

## Contexto

Notebook `listings.ipynb`. Problema de regresión: predecir el precio por noche de alojamientos de Airbnb en Río de Janeiro (dataset detallado de Inside Airbnb, 90 columnas). Las 12 secciones están escritas y el notebook corre de punta a punta sin errores ni warnings. Las secciones 1–4 son del compañero; el resto es del asistente.

## Formato de los archivos markdown

Sin quiebres de línea artificiales dentro del texto. Dejar que cada párrafo fluya como una sola línea; el word wrap del lector se encarga. Sí dejar líneas en blanco entre párrafos y entre ítems de lista. Regla: si el editor envolvería la línea en la pantalla, no insertar `\n` explícito.

## Reglas duras

- **No instalar paquetes.** Usar sólo lo declarado en `pyproject.toml`: numpy, pandas, scipy, scikit-learn, xgboost, optuna, matplotlib, seaborn.
- **Markdown mínimo**, sin títulos decorativos. Solo el texto que aporte.
- **Comentarios en código, siempre muy descriptivos.** Regla permanente: todo código nuevo que se agregue (cualquier sección) lleva comentarios que expliquen los conceptos de ML paso a paso — no sólo qué hace el código, sino por qué y qué significa cada pieza — para que el usuario pueda leerlos sin experiencia previa en ML. Son "úsalos y tiralos": el usuario los borra él mismo a medida que lee cada parte, así que no hay que ahorrar detalle. Las secciones 5–11 ya están así; mantener el mismo nivel de detalle en cualquier celda nueva.
- **Markdown de las secciones 5–12**: tono de observación, lacónico, sin explicar qué es cada algoritmo (el lector es un profesor, no alguien aprendiendo ML). Justificar decisiones específicas del dataset sí; explicar conceptos de libro no.
- **No sobrecomplicar.** Si el curso ya tiene una forma de hacer algo, copiarla.
- **Identificadores en código: sólo caracteres latinos simples (ASCII), sin tildes ni ñ.** Nombres de variables, funciones y settings van sin acentos (p.ej. `TAMANO_SUBSET`, no `TAMAÑO_SUBSET`) — comentarios y markdown en español siguen normales, esto es sólo para lo que el intérprete de Python lee como identificador.
- **No tocar las secciones 1 a 4** (son del compañero, incluida su celda de EDA/preprocesamiento), salvo para arreglar un error o warning existente — y ahí, lo mínimo indispensable.
- **Notebook debe correr sin warnings ni errores.** El bloque de imports tiene un único `warnings.filterwarnings("ignore")` que cubre todo (incluye un `UserWarning` informativo de `sklearn.utils.parallel` que aparece ~95 veces en el barrido de `max_features` de RF). Si se acota ese filtro en el futuro, correr el notebook completo después — puede destapar warnings que hoy están silenciados de más.
- **No entregar el cuaderno con el subsample de verificación activado.** La celda de Configuración define `TAMANO_SUBSET`; la celda de carga del dataset la usa (`if TAMANO_SUBSET is not None: ...`) para tomar sólo esa cantidad de filas. Para validar rápido: poner un número (p.ej. `TAMANO_SUBSET = 1000`). **Para entregar: dejarlo en `None`** (dataset completo). **Estado actual (2026-08-21): está en `None` y la corrida completa ya se hizo** sobre las 44.238 filas que quedan tras el corte de precio. Los resultados guardados en el notebook son los definitivos.
- **El notebook debe pasar `uvx ruff check trabajo_final/listings.ipynb` sin warnings.** No hace falta instalar ruff en el proyecto (rompería la regla de "no instalar paquetes") — `uvx ruff check ...` lo baja y corre sin tocar el entorno del notebook. Reglas activas: E, F, I, N, UP, PL (`pyproject.toml`). Antes de dar por cerrada una celda nueva, correr ese comando.
- **Imports: todos en la celda de imports** (justo después de la celda de Configuración, antes de la paleta de colores), en una sola lista ordenada alfabéticamente por módulo. No agregar un `import` suelto en una celda de más abajo aunque sólo se use ahí — si una celda nueva necesita un módulo no importado todavía, el import va arriba, no junto al uso.
- **Colores: sólo en la paleta de la celda de imports.** Si una celda nueva necesita un tono que no está en la paleta (p.ej. una gama para un gráfico), se define la constante ahí arriba (ver `NARANJA_CLARO`, `MAGENTA_CLARO`, `VIOLETA_OSCURO` para la gama de 11.2) en vez de escribir el hex suelto en la celda que lo usa.
- **Checkpointing (sección 5 en adelante)**: celda de infraestructura justo antes de 5.1 (`guardar_checkpoint`/`cargar_checkpoint`/`ya_evaluado`; el porqué de cada decisión de diseño está comentado ahí mismo, no lo repitas acá). Guarda `historial` tras cada `historial.append(...)`; en las 4 celdas más caras (SVR 6.1, barrido de RF 8.2, Optuna 9.3, Stacking 10.3) además persiste el objeto ya entrenado y saltea el reentrenamiento. Usa como mucho 5 `checkpoint_*.pkl` sueltos junto al notebook (gitignoreados). **Decisión pendiente:** si se deja en el cuaderno entregado o se saca antes de la entrega — no decidir sin preguntar.
- **Referencias bibliográficas: no agregar.** El enunciado las pide (`trabajo_final/README.md`), pero es una decisión explícita del usuario. No agregar la sección sin que lo pida de nuevo.
- **Mantener este archivo actualizado** ante cualquier cambio de diseño o hallazgo nuevo — es la única fuente de contexto persistente del trabajo.

## Helpers ya disponibles en el notebook (no redefinir)

- `construir_preprocesador()` — `ColumnTransformer` con `SimpleImputer(median)` + `StandardScaler` para numéricas, `OneHotEncoder(drop='first')` para categóricas, `passthrough` para binarias.
- `metricas(y_real, y_pred, nombre)` — devuelve MAE, RMSE, MAPE, R² en BRL.
- `evaluar(nombre, modelo, log_target=True)` — entrena, evalúa y devuelve fila lista para apilar. Por defecto entrena sobre `log(precio)` y mide en BRL. Pasar `log_target=False` únicamente en el experimento crudo vs log de 3.7. **Muta el objeto `modelo` que recibe** (le hace `.fit(X_train, objetivo_train)` adentro) — después de `historial.append(evaluar("X", modelo))`, `modelo` ya queda entrenado; no volver a fitearlo para leer coeficientes o importancias.
- `resultados` — lista con los 2 baselines. `exp` — lista con los dos OLS de 3.7 (crudo y log). `historial` — tabla viva donde se apila cada modelo nuevo (`historial.append(evaluar("Nombre", pipeline))`); la sección 11 arma `df_final` a partir de acá.
- `XGB_FIJOS` — dict con los kwargs fijos de XGBoost (`objective="reg:absoluteerror", random_state=SEMILLA, n_jobs=-1`), usado con `**XGB_FIJOS` en 9.3, 9.4 y 10.1.
- Constantes y paleta (`AZUL`, `MAGENTA`, `VERDE`, `NARANJA`, `CELESTE`, `GRIS`, `BASELINE`, `NARANJA_CLARO`, `MAGENTA_CLARO`, `VIOLETA_OSCURO`, `SEMILLA=42`) ya definidas arriba.
- `TAMANO_SUBSET` — setting de la celda de Configuración, `None` por defecto. Ver regla dura de arriba.

## Convención de target y `evaluar()`

Por decisión de 3.7 todos los modelos nuevos entrenan sobre `log(precio)`. Por eso `evaluar(name, model)` (sin flag) ya hace lo correcto. No repetir `log_target=True` en cada llamada.

## Métrica principal

**MAE en BRL** (medido siempre en reales, no en log). RMSE/MAPE/R² como secundarias.

## Antes de entregar

1. ~~Poner `TAMANO_SUBSET = None` y correr completo~~ **hecho** (2026-08-21). SVR fue el cuello de botella real: 107 min sólo esa celda.
2. ~~Confirmar que termina sin errores ni warnings~~ **hecho**: 56 celdas de código, 0 errores, 0 stderr, `execution_count` secuencial.
3. ~~Releer las secciones 8, 9, 11 y 12 contra los resultados de la corrida completa~~ **hecho**: se reescribió la 12 entera y se actualizaron los números de 2.5, 2.7, 2.8, 2.9, 2.11, 3.6, 3.7, 3.8, 4.2 y 4.3, que venían del scrape de marzo.
4. Decidir si el checkpointing se deja en el cuaderno entregado — ver la regla dura de arriba. **Sigue pendiente.**
5. Ejecutar la celda de distribución del error de la sección 12 (es nueva, quedó sin correr).

## Hallazgos de la corrida completa

- **Gana XGBoost (Optuna) con MAE 250,5 BRL**, empate técnico con Stacking (250,9), que le gana en R² y RMSE. La familia lineal queda en 294,3 y el baseline heurístico en 385.
- **La hipótesis del EDA se confirmó.** `latitude` es la feature más importante por permutación y `longitude` la tercera, pese a que la correlación lineal de `longitude` con el precio es −0,019. Es el argumento central del trabajo.
- **AdaBoost colapsa**: MAE 461,3, peor que el baseline heurístico, con MAE de train igual de malo (462,6). No sobreajusta, no aprende. Vale la pena dejarlo y explicarlo.
- **El árbol podado (334,8) queda peor que la regresión lineal.** El salto llega recién con Bagging y RF.
- **Las tres banderas de faltantes no aportaron nada** (importancia de permutación ≈ 0), ni tampoco `sin_reseñas`. Un árbol puede partir directo en `reviews_per_month == 0`. En Ridge, en cambio, `sin_reseñas` era el coeficiente más grande. Sirven para la familia lineal, no para la de árboles.
- **El error tiene forma de U**, no crece monótonamente con el precio: 46 % de error relativo en el quintil más barato, 23 % en el medio, 37 % en el más caro. Sólo el 45 % de las predicciones cae dentro de ±20 %.
- **En SVR ganó `C=1`**, así que los fits con `C=100` (la mayor parte de los 107 min) se descartaron. La grilla de la cátedra está pensada para 200 filas, no para 31k.

## Cambios sobre el notebook original

- Se agregaron tres banderas de ausencia (`sin_dormitorios`, `sin_baños`, `sin_camas`) al criterio de `sin_reseñas`. El cambio de scrape a junio llevó los faltantes de `bedrooms` de 123 a 6.446 casos (21,5 % de las filas). No aportaron señal, pero el párrafo de 3.1 que decía "menos del 1 % de las filas" era falso y había que corregirlo igual.
- k-NN, Voting y Stacking llevan ahora una frase que justifica su uso: su variante de regresión no se dio en clase (sólo la de clasificación), y se aclara que es el mismo algoritmo cambiando el paso final.
