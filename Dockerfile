FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 app \
    && useradd --create-home --shell /usr/sbin/nologin --uid 1001 --gid 1001 app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt pyproject.toml ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config/ ./config/
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

RUN mkdir -p /srv/data /srv/logs /srv/user_context && chown -R app:app /srv

USER app

EXPOSE 8440

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8440/health || exit 1

# MCP server mode. For CLI-only usage, exec into the container and
# run `council ...` directly instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8440"]
