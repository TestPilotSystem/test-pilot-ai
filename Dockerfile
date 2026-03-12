FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

ARG LLM_PROVIDER=ollama

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN if [ "$LLM_PROVIDER" = "groq" ]; then \
      pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt; \
    else \
      pip install --prefix=/install -r requirements.txt; \
    fi

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
ENV PATH="/usr/local/bin:$PATH"

COPY app/ ./app/

RUN mkdir -p temp_uploads vector_db



EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health-check')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
