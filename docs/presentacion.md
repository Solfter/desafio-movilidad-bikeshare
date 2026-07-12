# Guión de presentación (15 minutos)

> Objetivo: explicar el problema, la arquitectura técnica, hacer una **demo funcional** y presentar
> resultados y métricas. Distribución sugerida de tiempo entre paréntesis.

## 0. Portada e integrantes (0:30)
- Título: *Predicción de demanda de bicicletas — Desafío de Ciencia de Datos*.
- Nombres del equipo y rol de cada uno.

## 1. Problema y contexto (2:00)
- Un operador de *bikesharing* necesita **anticipar la demanda por hora** para redistribuir la flota,
  planificar mantenimiento y dimensionar la operación.
- Pregunta de negocio: **¿cuántas bicicletas se usarán la próxima hora dadas la fecha, el calendario y el clima?**
- Por qué importa: menos bicicletas sin stock, mejor servicio, menor costo operativo.

## 2. Datos y fuentes (2:00)
- **3 fuentes integradas** por el ETL:
  1. UCI Bike Sharing (Washington DC, horario, 2011–2012).
  2. **Open-Meteo** (clima histórico real, API sin clave) → aporta precipitación real.
  3. Feriados federales (librería `holidays`).
- Unión por `datetime`; **100% de match** con el clima; dataset final **17.158 filas × 42 columnas**.
- Mostrar 1 lámina con el diagrama de arquitectura (`docs/arquitectura.md`).

## 3. Arquitectura técnica (2:30)
- Pipeline **ETL modular** (extract → transform → features → load) → **Parquet**.
- **Modelado**: split temporal, comparación de 4 modelos, artefactos versionados.
- **Servicios**: API **FastAPI** (`/predict`) + **dashboard Plotly Dash**, desacoplados.
- **DevOps**: **Docker** (compose con 2 servicios) + **GitHub Actions** (ruff + pytest + build).
- Recalcar buenas prácticas: paquete instalable, tests, lint, reproducibilidad con `uv`.

## 4. Demo funcional (4:00) ⭐
1. **Dashboard** (`localhost:8050`):
   - KPIs y **exploración**: serie temporal, **heatmap hora×día** (doble peak de *commuting*), clima vs demanda.
   - Pestaña **Modelo**: predicho vs real e **importancia de variables**.
   - Pestaña **Predicción**: mover sliders (hora, clima…) y obtener la demanda estimada en vivo.
2. **API** (`localhost:8000/docs`): ejecutar un `POST /predict` desde Swagger.
   - Caso hora punta buen clima ≈ **395 bicis/h**; madrugada lluviosa ≈ **8 bicis/h**.

## 5. Resultados y métricas (2:00)
- Tabla comparativa (test, split temporal 80/20):

  | Modelo | MAE | RMSE | R² |
  |---|---:|---:|---:|
  | Baseline (lag 24 h) | 79.0 | 132.7 | 0.638 |
  | Regresión lineal | 61.4 | 90.8 | 0.831 |
  | Random Forest | 30.1 | 50.8 | 0.947 |
  | **XGBoost** | **27.3** | **43.7** | **0.961** |

- **XGBoost** explica el **96%** de la varianza; error medio ≈ 27 bicicletas/hora.
- Variables clave: **demanda reciente**, **día laboral**, **hora** (cíclica); el clima afina los extremos.

## 6. Cierre y trabajo futuro (0:30)
- Lo logrado: solución **end-to-end** (datos → modelo → API → dashboard → contenedor → CI).
- Futuro: reentrenamiento programado, más ciudades/estaciones, features de eventos, despliegue en la nube.

## 7. Preguntas (1:00)
- Anticipar: *¿por qué split temporal?*, *¿cómo evitan fuga de datos?*, *¿por qué XGBoost?*,
  *¿qué pasa si la API se cae?* (fallback local), *¿cómo escalarían a producción?*

---
### Checklist previo a la demo
- [ ] `uv sync` hecho y datos/modelo presentes (`make etl && make train` o artefactos versionados).
- [ ] API arriba: `uv run uvicorn bikeshare.api.main:app --port 8000`.
- [ ] Dashboard arriba: `uv run python -m bikeshare.dashboard.app`.
- [ ] Alternativa todo-en-uno: `docker compose up --build`.
- [ ] Tener el repo de GitHub abierto (Actions en verde).
