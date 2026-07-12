# 🚲 Desafío Movilidad — Predicción de demanda de bicicletas

Solución completa de ciencia de datos que **predice la demanda horaria de bicicletas** de un
sistema de *bikesharing*, integrando múltiples fuentes de datos, un pipeline ETL, modelos de
machine learning, una API y un dashboard interactivo, todo contenerizado y validado con CI/CD.

> Proyecto del **Desafío de Ciencia de Datos** (Duoc UC, 2026).

## 👥 Integrantes
- _(completar)_

## 🎯 Problema
Un operador de bicicletas compartidas necesita **anticipar cuántas bicicletas se usarán cada hora**
para redistribuir la flota, planificar mantenimiento y dimensionar la operación. Modelamos la
demanda horaria (`cnt`) en función de factores **temporales** (hora, día, estacionalidad, feriados)
y **meteorológicos** (temperatura, precipitación, humedad, viento).

## 🗃️ Fuentes de datos (3, integradas por ETL)
1. **UCI Bike Sharing Dataset** (Washington DC, 2011–2012, horario) — vía `ucimlrepo` (id=275).
2. **Open-Meteo Archive API** — clima histórico real (temperatura, precipitación, humedad, viento).
   Gratis y **sin clave**. Se une por `datetime`.
3. **Feriados federales de EE.UU.** — librería `holidays`, unido por fecha.

## 🏗️ Arquitectura
```
Fuentes → ETL (extract → transform/join → features → load) → Parquet
                                                              │
                          ┌───────────────────────────────────┼───────────────┐
                          ▼                                   ▼               ▼
                    Modelo ML (XGB)                       API FastAPI     Dashboard Dash
                    model.pkl + metrics                    /predict        (KPIs, series, mapa…)
```
Detalle en [`docs/arquitectura.md`](docs/arquitectura.md).

## 🚀 Cómo ejecutar

### Local (con `uv`)
```bash
uv sync                                      # instala dependencias (Python 3.13)
uv run python -m bikeshare.etl.pipeline      # 1) genera data/processed/bikeshare_features.parquet
uv run python -m bikeshare.models.train      # 2) entrena y guarda models/model.pkl + metrics.json
uv run uvicorn bikeshare.api.main:app --port 8000        # 3) API en http://localhost:8000/docs
uv run python -m bikeshare.dashboard.app                 # 4) dashboard en http://localhost:8050
```

### Con Docker
```bash
docker compose up --build     # API en :8000 y dashboard en :8050
```

## 🧪 Calidad
```bash
uv run ruff check .           # lint
uv run pytest                 # tests
```

## 📊 Resultados

Split **temporal** (80/20): 13.726 h de entrenamiento, 3.432 h de test. Métricas sobre el test:

| Modelo | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Baseline (demanda de hace 24 h) | 79.0 | 132.7 | 0.638 | 68.2% |
| Regresión lineal | 61.4 | 90.8 | 0.831 | 80.2% |
| Random Forest | 30.1 | 50.8 | 0.947 | 21.3% |
| **XGBoost (elegido)** | **27.3** | **43.7** | **0.961** | **21.7%** |

- El **XGBoost** explica el **96%** de la varianza de la demanda horaria, con un error medio de ~27 bicicletas.
- Variables más influyentes: **demanda reciente** (`lag_1h`, `lag_24h`), si es **día laboral**, y la **hora del día**
  (codificada de forma cíclica). El clima aporta de forma secundaria pero mejora los casos extremos (lluvia/frío).
- Ejemplos de predicción vía API: hora punta con buen clima ≈ **395 bicis/h**; madrugada lluviosa ≈ **8 bicis/h**.

## 📁 Estructura
```
src/bikeshare/{etl,models,api,dashboard}   # código fuente
data/{raw,interim,processed}               # datos (regenerables con el ETL)
models/                                    # model.pkl, metrics.json, importancias
notebooks/                                 # EDA
tests/                                     # pytest
docs/                                      # arquitectura + material de presentación
```

## 📄 Licencia
Uso académico.
