# DocuFlow API + Temporal worker (GPU OCR/LLM stay on host — see deploy/docker.env.example)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

EXPOSE 8022

# Default: API. Worker overrides command in docker-compose.
CMD ["uvicorn", "serving.workflow_api:app", "--host", "0.0.0.0", "--port", "8022"]
