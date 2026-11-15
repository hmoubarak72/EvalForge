.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help setup lint format typecheck test coverage migrate seed api dashboard pipeline evaluate scheduler export docker-up docker-down docker-seed check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Install the package with dev and PostgreSQL extras
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,postgres]"

lint:  ## Lint and check formatting (ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-format and fix lint issues
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:  ## Static type check (mypy)
	$(PYTHON) -m mypy

test:  ## Run the test suite (offline; no external API calls)
	$(PYTHON) -m pytest

coverage:  ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Lint, type check and test

migrate:  ## Apply database migrations (EVALFORGE_DATABASE_URL)
	$(PYTHON) -m alembic upgrade head

seed:  ## Reset the database and load the simulated demo
	$(PYTHON) scripts/seed_demo.py --reset

api:  ## Run the API at http://localhost:8000 (docs at /docs)
	$(PYTHON) -m uvicorn evalforge.api.app:app_factory --factory --reload --port 8000

dashboard:  ## Run the curation dashboard at http://localhost:8501
	$(PYTHON) -m streamlit run src/evalforge/ui/dashboard.py

pipeline:  ## Run the full pipeline once
	$(PYTHON) -m evalforge.cli pipeline

evaluate:  ## Evaluate the configured endpoint against the dataset
	$(PYTHON) -m evalforge.cli evaluate

scheduler:  ## Run the nightly scheduler in the foreground
	$(PYTHON) -m evalforge.cli scheduler

export:  ## Export the active dataset to data/eval_dataset.jsonl
	$(PYTHON) scripts/export_dataset.py --output data/eval_dataset.jsonl

docker-up:  ## Build and start PostgreSQL, API, pipeline worker and dashboard
	docker compose up --build -d

docker-seed:  ## Load the demo data into the Docker Compose stack
	docker compose --profile demo run --rm seed

docker-down:  ## Stop the stack (add ARGS=-v to drop volumes)
	docker compose down $(ARGS)
