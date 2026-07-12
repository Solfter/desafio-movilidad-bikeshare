# MEMORY — Bitácora del proyecto

> Bitácora de estado para dar **continuidad entre sesiones**. Actualizar al final de cada fase:
> qué quedó listo, decisiones tomadas y qué sigue.

## Resumen del proyecto
- **Caso:** predicción de **demanda horaria de bicicletas** (regresión) — movilidad urbana.
- **Dataset base:** UCI Bike Sharing (Washington DC, 2011–2012, horario), `ucimlrepo` id=275.
- **Enriquecimiento (3 fuentes):** UCI + Open-Meteo Archive (clima real, sin clave) + feriados US (`holidays`).
- **Stack:** Python 3.13 + `uv`, pandas/scikit-learn/xgboost, FastAPI, Plotly Dash, Docker, GitHub Actions.
- **Entregables:** encargo técnico + presentación (PPTX + guión 15 min).

## Decisiones clave
- Se descartó Kaggle a propósito (requiere credenciales); UCI y Open-Meteo no necesitan clave.
- Split **temporal** (sin shuffle) por ser serie de tiempo; validación con `TimeSeriesSplit`.
- `model.pkl` se **commitea** (pequeño) para que API/dashboard/CI corran sin reentrenar.

## Estado por fase
- [x] **Fase 0 — Bootstrap:** carpeta, git init, pyproject/uv, .gitignore, .dockerignore,
      esqueleto `src/bikeshare`, config.py, README y MEMORY iniciales. `uv sync` OK (Python 3.13).
- [x] **Fase 1 — ETL:** extract/transform/features/load/pipeline OK. Parquet final: **17.158 filas × 42 cols**,
      match clima **100%**, 0 nulos. Notebook `notebooks/01_eda.ipynb` generado y ejecutado.
- [x] **Fase 2 — ML:** XGBoost gana → **R²=0.961, MAE=27.3, RMSE=43.7** (baseline lag24h R²=0.64,
      LR R²=0.83, RF R²=0.947). Artefactos: model.pkl + metrics.json + feature_importances.csv.
- [x] **Fase 3 — API:** FastAPI /health, /model/info, /predict OK. Verificado en vivo (394.7 bicis en
      hora punta buen clima; 8.5 en madrugada lluviosa). Tests con TestClient. **14 tests pasan, ruff limpio.**
- [x] **Fase 4 — Dashboard:** Plotly Dash con 3 pestañas (Exploración con filtros, Modelo predicho-vs-real
      + importancias, Predicción what-if que consume la API con fallback local). Verificado en navegador (HTTP 200,
      KPIs correctos). Callback de predicción usa `config.API_URL`.
- [x] **Fase 5 — Docker:** Dockerfile (python:3.13-slim + uv) + docker-compose (api 8000 + dashboard 8050,
      healthcheck, API_URL). **Build y `up` verificados en vivo**: ambos contenedores corren, `api` healthy,
      `/health` y `/predict` responden (394.7), dashboard HTTP 200. Se corrigió una condición de carrera:
      ambos servicios compartían `image: bikeshare:latest` con `build:` propio → competían al exportar el
      mismo tag ("already exists"). Fix: solo `api` tiene `build:`; `dashboard` reutiliza esa imagen ya
      etiquetada (ver comentario en `docker-compose.yml`).
- [x] **Fase 6 — CI/CD:** `.github/workflows/ci.yml` con uv sync + ruff + pytest + docker build.
      Se pondrá verde al hacer el primer push (Fase 8).
- [x] **Fase 7 — Docs + presentación:** README con tabla de métricas, `docs/arquitectura.md` (Mermaid),
      `docs/presentacion.md` (guión 15 min) y `docs/presentacion.pptx` (9 slides, paleta verde). Deck valida
      con python-pptx. QA visual pendiente (no hay LibreOffice/poppler en la máquina; abrir en PowerPoint).
- [ ] **Fase 8 — Publicar:** commit + repo GitHub + push (**requiere permiso explícito del usuario**).

## Pendientes / notas
- Verificar `uv sync` en la máquina (Python 3.13).
- Nombres de integrantes para el README (placeholder por ahora).
- Confirmar permiso antes de publicar en GitHub (Fase 8).
