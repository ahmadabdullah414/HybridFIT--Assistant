# Hugging Face Spaces (Docker SDK) deployment for the HybridFit Assistant
# API. HF Spaces expects the container to listen on port 7860.
FROM python:3.11-slim

WORKDIR /app

# build-essential/cmake: llama-cpp-python falls back to compiling from
# source if no prebuilt wheel matches this exact platform/Python combo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Model files live under flutter/ (the same files the mobile app bundles/
# downloads) so there's a single source of truth, not a duplicated copy.
COPY flutter/llm/fitbot-q4_k_m.gguf flutter/llm/fitbot-q4_k_m.gguf
COPY flutter/nlu flutter/nlu
COPY api api

WORKDIR /app/api
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
