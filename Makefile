# Exoplanet Transit Hunter — task runner. Run `make help` for the list.
# Quickstart:  make setup && make demo

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV      ?= uv
RUN     := $(UV) run
LIMIT   ?= 800
TESS_LIMIT ?= 400
WORKERS ?= 8
API_PORT ?= 8000
UI_PORT  ?= 8501
TARGETS ?=

.PHONY: help setup setup-full lock data-labels data-download data-status data-process data-tess \
        data data-synth train-gbm export-model \
        train-cnn train evaluate warm-cache api ui demo hunt test test-network lint format \
        typecheck check docker-build docker-up docker-down clean clean-data

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m [VAR=value]\n\n"} \
	  /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables: LIMIT=$(LIMIT) WORKERS=$(WORKERS) API_PORT=$(API_PORT) UI_PORT=$(UI_PORT)"

# ---------------------------------------------------------------- setup
setup: ## Create .venv (Python 3.11) with all core + dev deps, install pre-commit hooks
	$(UV) python install 3.11
	$(UV) sync --extra dev
	@test -f .env || cp .env.example .env
	-$(RUN) pre-commit install

setup-full: ## Like setup, plus stretch deps (transitleastsquares, batman-package)
	$(UV) python install 3.11
	$(UV) sync --all-extras
	@test -f .env || cp .env.example .env
	-$(RUN) pre-commit install

lock: ## Regenerate uv.lock
	$(UV) lock

# ---------------------------------------------------------------- data
data-labels: ## Download + map Kepler KOI labels from the NASA Exoplanet Archive
	$(RUN) python scripts/build_dataset.py --stage labels

data-download: ## Download Kepler + TESS light curves (night before!). LIMIT=N TESS_LIMIT=N PURGE=1
	$(RUN) python scripts/build_dataset.py --stage download --limit $(LIMIT) --tess-limit $(TESS_LIMIT) \
	  --workers $(WORKERS) $(if $(PURGE),--purge-fits,)

data-status: ## Progress of the light-curve download
	$(RUN) python scripts/build_dataset.py --stage status

data-process: ## Cached light curves -> views/features -> data/processed (offline)
	$(RUN) python scripts/build_dataset.py --stage process --workers $(WORKERS)

data-tess: ## Optional TESS domain-shift validation set (never trained on)
	$(RUN) python scripts/build_dataset.py --stage tess --tess-limit 150 --workers $(WORKERS)

data: ## Full dataset: labels -> download -> clean/detrend/fold -> data/processed. LIMIT=N
	$(RUN) python scripts/build_dataset.py --stage all --limit $(LIMIT) --workers $(WORKERS)

data-synth: ## Fallback: inject synthetic PC/EB signals (batman if installed, else trapezoids)
	$(RUN) python scripts/build_dataset.py --stage synth

# ---------------------------------------------------------------- models
train-gbm: ## Train the LightGBM baseline
	$(RUN) python scripts/train.py --model gbm

train-cnn: ## Train the dual-branch CNN
	$(RUN) python scripts/train.py --model cnn

train: ## Train GBM + CNN, evaluate, write models/metrics.json + models/results.md
	$(RUN) python scripts/train.py --model both

evaluate: ## Re-evaluate saved checkpoints (val / test / tess)
	$(RUN) python scripts/train.py --evaluate-only

export-model: ## Copy current checkpoints + metrics to models/final (committed for the demo)
	$(RUN) python scripts/train.py --evaluate-only --export-final

# ---------------------------------------------------------------- serving
warm-cache: ## Pre-fetch + pre-compute demo targets into data/demo_cache
	$(RUN) python scripts/warm_demo_cache.py

api: ## Run the FastAPI backend (reload mode)
	$(RUN) uvicorn transit_hunter.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

ui: ## Run the Streamlit frontend
	TH_API_URL=http://localhost:$(API_PORT) $(RUN) streamlit run frontend/app.py --server.port $(UI_PORT)

demo: ## Run API + UI together (Ctrl-C stops both)
	@trap 'kill 0' EXIT; \
	$(RUN) uvicorn transit_hunter.api.main:app --host 0.0.0.0 --port $(API_PORT) & \
	sleep 2; \
	TH_API_URL=http://localhost:$(API_PORT) $(RUN) streamlit run frontend/app.py \
	  --server.port $(UI_PORT) --server.headless true; \
	wait

hunt: ## Batch-scan targets: make hunt TARGETS="TIC 1 TIC 2"
	$(RUN) python scripts/hunt.py $(TARGETS)

# ---------------------------------------------------------------- quality
test: ## Run offline tests (no network)
	$(RUN) pytest

test-network: ## Run tests that hit MAST / Exoplanet Archive
	$(RUN) pytest -m network

lint: ## Ruff lint
	$(RUN) ruff check src tests scripts frontend

format: ## Ruff format + autofix
	$(RUN) ruff format src tests scripts frontend
	$(RUN) ruff check --fix src tests scripts frontend

typecheck: ## mypy on src/
	$(RUN) mypy

check: lint typecheck test ## Lint + typecheck + tests (what CI runs)

# ---------------------------------------------------------------- docker
docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start API + UI in Docker
	docker compose up -d

docker-down: ## Stop Docker services
	docker compose down

# ---------------------------------------------------------------- cleanup
clean: ## Remove caches and build artifacts (keeps data/ and models/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data: ## Remove interim + processed data (keeps raw downloads and demo cache)
	rm -rf data/interim/* data/processed/*
	touch data/interim/.gitkeep data/processed/.gitkeep
