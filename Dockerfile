FROM apache/superset:latest

USER root

# Ставим пакет глобально в окружение контейнера, игнорируя ограничения venv
RUN . /app/.venv/bin/activate && \
    uv pip install --no-cache-dir "clickhouse-connect>=0.13.0"

USER superset