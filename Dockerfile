FROM python:3.12-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/huggingface \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    HF_HUB_DISABLE_XET=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app
COPY pyproject.toml uv.lock ./
ARG PYPI_FILE_MIRROR=https://files.pythonhosted.org/
ARG UV_HTTP_TIMEOUT=120
# A mirror can serve the same wheels when the default CDN is slow. uv still verifies lockfile hashes.
RUN --mount=type=cache,target=/root/.cache/uv \
    python -c "from os import environ; from pathlib import Path; lock = Path('uv.lock'); lock.write_text(lock.read_text().replace('https://files.pythonhosted.org/', environ['PYPI_FILE_MIRROR'].rstrip('/') + '/'))" \
    && uv sync --locked --no-dev --no-install-project

COPY app ./app
COPY data/processed ./data/processed
# Download the model and seed both Chroma collections before first startup.
RUN .venv/bin/python -c "from pathlib import Path; from app.services.indexer import SearchIndex; from app.services.normalizer import Normalizer; data = Path('data/processed'); SearchIndex(data, Path('.chroma'), Normalizer(data))"

FROM python:3.12-slim-trixie

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    RERANKER_MODEL=off \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY --from=build /opt/huggingface /opt/huggingface
COPY --from=build /app/app ./app
COPY --from=build /app/data ./data
COPY --from=build --chown=10001:10001 /app/.chroma ./.chroma

USER 10001:10001
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2).close()"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
