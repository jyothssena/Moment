FROM python:3.11-slim

WORKDIR /app

ARG GIT_SHA=local
ARG MODEL_MODE=stub

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────────────────────
# Install from requirements.txt first (shared with local dev),
# then layer API-specific packages that aren't in requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       fastapi>=0.111.0 \
       uvicorn[standard]>=0.30.0 \
       pydantic>=2.7.0

# ── Copy model files ─────────────────────────────────────────────────────────
# These live in subdirectories in the repo but the API imports them as bare
# modules (e.g. `from model_interface import ...`), so they must be flat in
# /app. The build context is the repo root (docker build --file api/Dockerfile .)
#
#   Repo path                              → Container path
#   models/aggregator.py                   → /app/aggregator.py
#   models/decomposing_agent.py            → /app/decomposing_agent.py
#   models/compatibility_agent.py          → /app/compatibility_agent.py
#   cicd_pipeline/model_interface.py       → /app/model_interface.py
#   cicd_pipeline/validate_model.py        → /app/validate_model.py
#   cicd_pipeline/rollback.py              → /app/rollback.py
#   cicd_pipeline/notifications.py         → /app/notifications.py
#   recommendation_agent.py  (root)        → /app/recommendation_agent.py
#   tools.py                 (root)        → /app/tools.py
#   run_api.py               (root)        → /app/run_api.py
#   verify_deployment.py     (root)        → /app/verify_deployment.py

# From models/
COPY models/aggregator.py              .
COPY models/decomposing_agent.py       .
COPY models/compatibility_agent.py     .
COPY models/bradley_terry.py           .

# From cicd_pipeline/
COPY cicd_pipeline/model_interface.py  .
COPY cicd_pipeline/validate_model.py   .
COPY cicd_pipeline/rollback.py         .
COPY cicd_pipeline/notifications.py    .

# From repo root (already at the correct level)
COPY recommendation_agent.py           .
COPY tools.py                          .
COPY run_api.py                        .
COPY verify_deployment.py              .

# ── Copy API package ─────────────────────────────────────────────────────────
COPY api/ ./api/

# ── Copy processed data (optional — stub mode doesn't need it) ───────────────
COPY data/processed/           ./data/processed/

# ── Environment ──────────────────────────────────────────────────────────────
ENV MODEL_MODE=${MODEL_MODE}
ENV GIT_SHA=${GIT_SHA}
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
