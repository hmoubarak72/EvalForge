FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so source changes do not invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[postgres]"

COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 evalforge \
    && mkdir -p /app/data && chown -R evalforge:evalforge /app
USER evalforge

EXPOSE 8000 8501

# Default command: the API. docker-compose overrides it for the other services.
CMD ["uvicorn", "evalforge.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
