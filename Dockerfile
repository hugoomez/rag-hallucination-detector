# Single image for both services (API + Gradio), per ADR-018.
# docker-compose runs this one image two ways (different command/port), so the
# heavy torch/transformers layers are built and cached once. See docker-compose.yml.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf-cache

# Dependencies first, for layer caching. CPU-only torch via the index URL in the file.
# Generous timeout/retries: the CPU torch wheel is large and its mirror can be slow.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout 180 --retries 5 -r requirements-docker.txt

# Force classic HTTP model downloads. The hf_xet backend (auto-installed with
# huggingface_hub) can hang inside containers; this makes first-start downloads reliable.
ENV HF_HUB_DISABLE_XET=1

# Code + data assets. Models are NOT baked in -- they download at first startup
# into HF_HOME (mounted as a named volume by compose, so restarts reuse them).
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY data/corpus/ ./data/corpus/
# The custom demo, served by the API process itself at "/" (ADR-019). Fonts are vendored
# in here, so the design renders identically with no network access.
COPY frontend/ ./frontend/

EXPOSE 8000 7860

# Default command (the API). compose overrides it for the Gradio service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
