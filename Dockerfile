# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build dependencies into a venv
#
# Separated so compilers and headers stay out of the final image. chromadb and
# tiktoken pull native wheels; if a wheel is missing for the target platform,
# pip falls back to building from source and needs gcc.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone, before the source, so this layer is cached and only re-runs
# when dependencies actually change. Copying source first would invalidate the
# cache on every code edit and reinstall everything each build.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    MODE=ui

# Non-root. A container running as root that is exposed to the internet gives
# an attacker who finds a hole in the app a much better starting position.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

# The pre-built index ships with the image. The alternative — embedding the
# corpus at container start — costs money and several minutes on every restart,
# including every autoscale event and every Spaces cold boot. Build it locally
# with `python src/ingest.py`, then build the image.
COPY --chown=appuser:appuser .chroma/ ./.chroma/

RUN chmod +x entrypoint.sh
USER appuser

EXPOSE 8501

# Streamlit's own health endpoint. Reports the server is up; it does not check
# that the index loaded, which is what /health on the API is for.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/_stcore/health" || exit 1

ENTRYPOINT ["./entrypoint.sh"]