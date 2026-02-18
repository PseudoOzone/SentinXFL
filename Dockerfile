# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -e ".[dev]" || true

COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# ── Dashboard build ───────────────────────────────────────────
FROM node:20-alpine AS dashboard-builder

WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY dashboard/ ./
RUN npm run build

# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r sentinxfl && useradd -r -g sentinxfl sentinxfl

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy source
COPY src/ src/
COPY pyproject.toml ./
COPY .env.example .env.example

# Copy built dashboard
COPY --from=dashboard-builder /dashboard/dist dashboard/dist/

# Create required directories with correct ownership
RUN mkdir -p data/datasets data/processed data/uploads models/checkpoints logs/audit chroma_db \
    && chown -R sentinxfl:sentinxfl /app

# Drop to non-root
USER sentinxfl

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Production entrypoint — no reload, limited workers
CMD ["python", "-m", "uvicorn", "sentinxfl.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--no-access-log", \
     "--limit-concurrency", "100", "--timeout-keep-alive", "30"]
