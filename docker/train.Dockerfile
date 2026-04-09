# Training container
# Base image: PyTorch 2.0 + CUDA 11.7 (matches RTX 2060)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
# torch already included in base image, so we skip it here
COPY docker/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy source code
COPY training/    ./training/
COPY data_pipeline/ ./data_pipeline/
COPY scripts/     ./scripts/
COPY serving/     ./serving/

# Data & model output are mounted via Docker volume at runtime:
#   data/   → /app/data
#   models/ → /app/models

CMD ["python", "-m", "training.train"]
