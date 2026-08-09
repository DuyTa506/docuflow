# DocuFlow API + Temporal worker (GPU OCR/LLM stay on host — see deploy/docker.env.example)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libpq5/curl: DB driver + healthcheck.
# pandoc: DOCX export via OMML (utils/markdown_pandoc.py, utils/math_omml.py).
#   Absent, the export silently drops to the python engine and loses formulas.
# libreoffice-writer + fonts: DOC/DOCX conversion (utils/soffice.py). The
#   -writer package alone is ~400MB smaller than full libreoffice and is all
#   the Writer-only conversion path needs.
# gcc + libc6-dev: only used if AF_UNIX sockets are blocked at runtime, in
#   which case utils/soffice.py compiles an LD_PRELOAD shim on first use.
#   Normal containers never hit this, but the build must not fail if they do.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    pandoc \
    libreoffice-writer \
    fonts-liberation \
    fonts-dejavu-core \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

# Fail the build rather than ship an image that silently loses DOCX export or
# .doc ingestion — the exact gap this Dockerfile used to have.
RUN python - <<'PY'
import sys

from utils.native_deps import NATIVE_DEPS, missing_native_dependencies

missing = missing_native_dependencies()
for dep in missing:
    print(f"MISSING {dep.name}: {dep.impact}", file=sys.stderr)
if missing:
    sys.exit(1)
print("native deps ok:", ", ".join(dep.name for dep in NATIVE_DEPS))
PY

EXPOSE 8022

# Default: API. Worker overrides command in docker-compose.
CMD ["uvicorn", "serving.workflow_api:app", "--host", "0.0.0.0", "--port", "8022"]
