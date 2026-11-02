FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reproducible dependency installation from pyproject.toml
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY app ./app
COPY models ./models

EXPOSE 8000

# Render provides the port to bind to via the $PORT environment variable - default to
# 8000 for local/manual runs (e.g. docker run -p 8000:8000 ...)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]