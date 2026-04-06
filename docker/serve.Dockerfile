# Serving container (FastAPI inference)
# Base image: PyTorch 2.0 + CUDA 11.7 (GPU inference)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source code
COPY serving/     ./serving/
# model 架構（inference 時需要 load model class）
COPY training/    ./training/
# RAG explanation module
COPY rag/         ./rag/

# Model weights are mounted via Docker volume at runtime:
#   models/ → /app/models

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
