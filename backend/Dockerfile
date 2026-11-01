# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    ASHARE_ENABLE_SCHEDULER=0

COPY --from=uv /uv /uvx /bin/

WORKDIR /app/backend

# Install dependencies first so application edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY configs ./configs
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

ENV PATH="/app/backend/.venv/bin:$PATH" \
    ASHARE_DATABASE_URL="sqlite:////app/backend/data/app.db"

RUN mkdir -p /app/backend/data

EXPOSE 8000

# One worker only: the in-process APScheduler must not be duplicated.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn --factory app.api.app:create_app --host 0.0.0.0 --port 8000 --workers 1"]
