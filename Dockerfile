# syntax=docker/dockerfile:1
#
# Multi-stage build: dependencies are installed into a virtualenv in a
# "builder" stage (which has compilers available as a fallback for any
# package without a prebuilt wheel); the final image only copies the venv,
# so build tools never ship.

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch first, so sentence-transformers does not pull the
# multi-GB CUDA build from PyPI.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0"

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1: OpenMP runtime required by xgboost's Linux wheels
# (previously provided implicitly by build-essential).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Copy the application code (see .dockerignore for exclusions)
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Runs behind Caddy (deploy/Caddyfile), so trust its X-Forwarded-* headers
CMD ["sh", "-c", "uvicorn api_main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
