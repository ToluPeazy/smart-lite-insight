.PHONY: help dev build up down logs seed train test lint fmt pre-commit clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Local Development ──

dev: ## Run API + dashboard locally (no Docker)
	@echo "Starting API on :8000 and Dashboard on :8501..."
	@echo "Press Ctrl+C to stop"
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload &
	streamlit run dashboard/app.py --server.port 8501 --server.headless true

test: ## Run all tests with coverage
	pytest -v --cov=src --cov=seed --cov-report=term-missing

lint: ## Check formatting and linting (black + ruff)
	black --check --diff src/ dashboard/ seed/ tests/
	ruff check src/ dashboard/ seed/ tests/

fmt: ## Auto-format code (black + ruff --fix)
	black src/ dashboard/ seed/ tests/
	ruff check --fix src/ dashboard/ seed/ tests/

pre-commit: ## Install and run pre-commit hooks
	pre-commit install
	pre-commit run --all-files

train: ## Train the anomaly detection model
	python -m src.train

train-compare: ## Train and compare IF vs LOF
	python -m src.train --compare

seed-data: ## Generate 7 days of synthetic data
	python -m seed.replayer --days 7

ingest: ## Bulk load UCI dataset into SQLite
	python -m src.ingest

# ── Docker ──

build: ## Build Docker images
	docker compose build

up: ## Start all services (API + Dashboard)
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Tail service logs
	docker compose logs -f

seed: ## Generate synthetic data in Docker
	docker compose run --rm seed

# ── Cleanup ──

clean: ## Remove generated files (DB, models, cache)
	rm -f data/processed/energy.db
	rm -f models/*.joblib
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
