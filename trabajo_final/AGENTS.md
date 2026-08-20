# AGENTS.md — Trabajo Práctico Final (Airbnb Río de Janeiro)

## Contexto

Notebook `listings.ipynb`. Problema de regresión: predecir el precio por noche de alojamientos de Airbnb en Río de Janeiro (dataset detallado de Inside Airbnb, 90 columnas). Las 12 secciones están escritas y el notebook corre de punta a punta sin errores ni warnings. Las secciones 1–4 son del compañero; el resto es del asistente.

## Formato de los archivos markdown

Sin quiebres de línea artificiales dentro del texto. Dejar que cada párrafo fluya como una sola línea; el word wrap del lector se encarga. Sí dejar líneas en blanco entre párrafos y entre ítems de lista. Regla: si el editor envolvería la línea en la pantalla, no insertar `\n` explícito.

## Reglas duras

- **No instalar paquetes.** Usar sólo lo declarado en `pyproject.toml`: numpy, pandas, scipy, scikit-learn, xgboost, optuna, matplotlib, seaborn.
- **Markdown mínimo**, sin títulos decorativos. Solo el texto que aporte.
- **Comentarios en código, siempre muy descriptivos.** Regla permanente, no sólo para lo ya escrito: todo código nuevo que se agregue (cualquier sección) lleva comentarios que expliquen los conceptos de ML paso a paso — no sólo qué hace el código, sino por qué y qué significa cada pieza — para que el usuario pueda leerlos sin experiencia previa en ML. Son "úsalos y tiralos": el usuario los borra él mismo a medida que lee cada parte, así que no hay que ahorrar detalle pensando en que van a quedar para siempre. Las secciones 5–11 ya están así (ver sesión donde se aplicó); mantener el mismo nivel de detalle en cualquier celda nueva.
- **Markdown de las secciones 5–12**: tono de observación, lacónico, sin explicar qué es cada algoritmo (el lector es un profesor, no alguien aprendiendo ML). Justificar decisiones específicas del dataset sí; explicar conceptos de libro no.
- **No sobrecomplicar.** Si el curso ya tiene una forma de hacer algo, copiarla.
- **Identificadores en código: sólo caracteres latinos simples (ASCII), sin tildes ni ñ.** Nombres de variables, funciones y settings van sin acentos (p.ej. `TAMANO_SUBSET`, no `TAMAÑO_SUBSET`) — comentarios y markdown en español siguen normales, esto es sólo para lo que el intérprete de Python lee como identificador.
- **No tocar las secciones 1 a 4** (son del compañero, incluida su celda de EDA/preprocesamiento), salvo para arreglar un error o warning existente — y ahí, lo mínimo indispensable.
- **Notebook debe correr sin warnings ni errores.** El bloque de imports tiene un único `warnings.filterwarnings("ignore")` que cubre todo (incluye un `UserWarning` informativo de `sklearn.utils.parallel` que aparece ~95 veces en el barrido de `max_features` de RF). Si se acota ese filtro en el futuro, correr el notebook completo después — puede destapar warnings que hoy están silenciados de más.
- **No entregar el cuaderno con el subsample de verificación activado.** La celda de Configuración (la primera del cuaderno, ex-`pip install optuna`) define `TAMANO_SUBSET = None`; la celda de carga del dataset la usa (`if TAMANO_SUBSET is not None: ...`) para tomar sólo esa cantidad de filas. Para validar rápido: poner un número (p.ej. `TAMANO_SUBSET = 1000`). **Para entregar: dejarlo en `None`** (así se entrena sobre el dataset completo).
- **El notebook debe pasar `uvx ruff check trabajo_final/listings.ipynb` sin warnings.** No hace falta instalar ruff en el proyecto (rompería la regla de "no instalar paquetes") — `uvx ruff check ...` lo baja y corre sin tocar el entorno del notebook. Reglas activas: E, F, I, N, UP, PL (`pyproject.toml`). Antes de dar por cerrada una celda nueva, correr ese comando.
- **Imports: todos en la celda de imports** (justo después de la celda de Configuración, antes de la paleta de colores), en una sola lista ordenada alfabéticamente por módulo. No agregar un `import` suelto en una celda de más abajo aunque sólo se use ahí — si una celda nueva necesita un módulo no importado todavía, el import va arriba, no junto al uso.
- **Colores: sólo en la paleta de la celda de imports.** Si una celda nueva necesita un tono que no está en la paleta (p.ej. una gama para un gráfico), se define la constante ahí arriba (ver `NARANJA_CLARO`, `MAGENTA_CLARO`, `VIOLETA_OSCURO` para la gama de 11.2) en vez de escribir el hex suelto en la celda que lo usa.
- **Checkpointing (sección 5 en adelante)**: hay una celda de infraestructura (justo antes de 5.1) que define `guardar_checkpoint`/`cargar_checkpoint`/`ya_evaluado` y restaura `historial` si ya existe un checkpoint — pensada para que una desconexión a mitad de la corrida larga no tire horas de cómputo. Guarda `historial` completo después de cada `historial.append(...)` de las secciones 5–10, y además, en las 4 celdas más caras (SVR 6.1, barrido de RF 8.2, Optuna 9.3, Stacking 10.3), persiste el objeto ya entrenado y saltea el reentrenamiento si ya está hecho. Siempre usa unos pocos `checkpoint_*.pkl` sueltos junto al notebook (gitignoreados) — máximo 5 archivos (historial, grid_svr, rf_sweep, optuna_study, stacking), así que no se armó una carpeta aparte. Decisión explícita del usuario, por no depender de una cuenta de Google ni de la autorización interactiva de montar Drive. Como el directorio de trabajo al abrir el notebook varía según el entorno (Jupyter/Colab lo ponen en la carpeta del notebook; la extensión de Jupyter de VS Code por defecto usa la raíz del repo), la celda arma la ruta base con `Path.cwd().name == "trabajo_final"` para cubrir ambos casos sin depender de dónde se lanzó la sesión. Contrapartida asumida: en Colab ese disco es el de la VM, así que una reasignación de runtime (no sólo una desconexión) pierde los checkpoints de esa sesión. Decisión pendiente: si se deja en el cuaderno entregado o se saca antes de la entrega (agrega guardas que un lector no necesita para seguir el análisis) — no decidir sin preguntar.
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

1. Confirmar que `TAMANO_SUBSET` está en `None` en la celda de Configuración y correr el notebook completo (estimado 4–9 horas; SVR, el barrido de RF y Stacking son los cuellos de botella). Si el runtime es prohibitivo, recortar en este orden: trials de Optuna → RF del sweep → RF del stacking → grilla de SVR.
2. Confirmar que termina sin errores ni warnings.
3. Releer las secciones 8, 9, 11 y 12 contra los resultados reales (ver hallazgos abajo) — varias afirmaciones se escribieron antes de tener una corrida completa.
4. Decidir si el checkpointing se deja en el cuaderno entregado o se saca — ver la regla dura de arriba.

## Hallazgos pendientes de re-verificación contra la corrida real

- **Importancia de barrio (secciones 8 y 12)**: afirman que las dummies de barrio están entre las features más importantes de RF. En una corrida con el subsample de verificación (n chico) esto puede ser **imposible de verificar**: si el umbral de 200 apariciones mínimas de 3.3 deja a `neighbourhood_cleansed` con una sola categoría, `OneHotEncoder(drop='first')` no genera ninguna columna para esa variable y no puede aparecer como importante. Confirmar que la afirmación se sostiene con la corrida real.
- En general, cualquier frase de las secciones 8–12 que explique *por qué* gana un modelo (no sólo *que* gana) conviene releerla después de la corrida real.
