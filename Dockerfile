# One image serves both the API and the Streamlit UI (docker-compose picks the command).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# build-essential: some astro wheels (batman-package) compile C. curl: container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.4"

WORKDIR /app

# Dependency layer first, so code edits don't reinstall everything.
COPY pyproject.toml README.md ./
COPY uv.lock* ./
RUN mkdir -p src/transit_hunter && touch src/transit_hunter/__init__.py \
    && uv sync --no-install-project --no-dev

COPY src ./src
COPY configs ./configs
COPY frontend ./frontend
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN uv sync --no-dev

# Models and the demo cache are mounted as volumes (see docker-compose.yml) so the image stays small.
RUN mkdir -p data/raw data/interim data/processed data/demo_cache data/cache models

EXPOSE 8000 8501
CMD ["uvicorn", "transit_hunter.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
