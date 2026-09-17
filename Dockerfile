# Debian slim with Python 3.13, the same version as conda env used in develepoment
FROM python:3.13-slim

# No .pyc files; print logs immediately instead of buffering; no pip download cache in the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# All following paths are relative to /app inside the image. Settings' "data/test_set" resolves agains it.
WORKDIR /app

# Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code and test set last: editing code only rebuilds from here down.
COPY src/ ./src/
COPY data/test_set/ ./data/test_set/

# Documents the port. Actually publishing it to your machine happens in compose.
EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: inside a container, "localhost" is unreachable from outside the container.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
