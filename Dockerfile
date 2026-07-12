# Imagen única para API y dashboard (se elige el servicio por `command`).
FROM python:3.13-slim

# Copia los binarios de uv desde la imagen oficial.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Instalar dependencias (capa cacheable). Se necesita el código y el README
#    porque el proyecto es un paquete instalable (hatchling).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# 2) Copiar artefactos requeridos en runtime.
COPY models ./models
COPY data/processed ./data/processed

# El venv de uv queda en /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8050

# Por defecto levanta la API; el dashboard sobreescribe el `command` en compose.
CMD ["uvicorn", "bikeshare.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
