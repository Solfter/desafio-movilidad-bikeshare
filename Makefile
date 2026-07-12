# Atajos del proyecto. En Windows sin `make`, ejecutar directamente los comandos `uv run ...`.

.PHONY: sync etl train api dashboard test lint docker-up docker-down all

sync:
	uv sync

etl:
	uv run python -m bikeshare.etl.pipeline

train:
	uv run python -m bikeshare.models.train

api:
	uv run uvicorn bikeshare.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	uv run python -m bikeshare.dashboard.app

test:
	uv run pytest

lint:
	uv run ruff check .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

all: sync etl train test
