#!/usr/bin/env bash
# Runs either the Streamlit UI (MODE=ui, default) or the FastAPI backend
# (MODE=api). One image, two services — simpler than maintaining two Dockerfiles
# that differ by one line.
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "  docker run -e OPENAI_API_KEY=sk-... ..." >&2
  echo "  On HF Spaces, add it under Settings -> Secrets." >&2
  exit 1
fi

# Fail at startup rather than on the first user query. A container that boots
# healthy and then errors on every request is much harder to diagnose than one
# that refuses to start with a clear reason.
if [ ! -f "/home/appuser/app/.chroma/chroma.sqlite3" ]; then
  echo "ERROR: no index found at .chroma/" >&2
  echo "  The image is built with a pre-built index. Run 'python src/ingest.py'" >&2
  echo "  locally, then rebuild the image." >&2
  exit 1
fi

echo "starting mode=${MODE} port=${PORT}"

case "${MODE}" in
  ui)
    exec streamlit run src/app.py \
      --server.port="${PORT}" \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false
    ;;
  api)
    exec uvicorn api:app \
      --app-dir src \
      --host 0.0.0.0 \
      --port "${PORT}"
    ;;
  *)
    echo "ERROR: MODE must be 'ui' or 'api', got '${MODE}'" >&2
    exit 1
    ;;
esac