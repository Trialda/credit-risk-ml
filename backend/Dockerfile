# ---- Base: system deps + Python deps, shared by dev and production ----
FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -r requirements.txt

# ---- Dev target: local Docker Compose, hot-reload, full source tree ----
FROM base AS dev
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- Production target: no reload, no tests, no dev tooling ----
FROM base AS production
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]