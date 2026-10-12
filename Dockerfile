# syntax=docker/dockerfile:1.7
# ---------- build stage: resolve deps into an isolated venv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[local-models]"

# Optionally bake the ONNX embedding + reranker models into the image (no cold-start download).
ARG PRELOAD_MODELS=1
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "\
from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"; \
    else mkdir -p /opt/models; fi

# ---------- runtime stage: slim, non-root ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/models HF_HUB_DISABLE_TELEMETRY=1 \
    DATA_DIR=/app/data EMBEDDING_PROVIDER=fastembed RERANKER=cross-encoder ENV=prod
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models
WORKDIR /app
RUN mkdir -p /app/data && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=4)"
# Single worker: agent checkpoints and pending approvals live in-process (see README > Scaling).
CMD ["sh", "-c", "uvicorn finsight.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
