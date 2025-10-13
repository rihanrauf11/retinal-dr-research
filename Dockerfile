# Dockerfile for Diabetic Retinopathy Classification with RETFound + LoRA
# Optimized for NVIDIA RTX 5090 (32GB VRAM) on Vast.ai
# Base: PyTorch 2.1.0 with CUDA 12.1 support

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace/dr-classification

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY docs/ docs/
COPY CLAUDE.md .
COPY README.md .

# Create directories for volume mounts
# These will be mounted from Vast.ai persistent storage
RUN mkdir -p /data /models /results

# Set Python path
ENV PYTHONPATH=/workspace/dr-classification:$PYTHONPATH

# Expose port for Jupyter (optional)
EXPOSE 8888

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]

# Build instructions:
# docker build -t your-dockerhub-username/dr-retfound-lora:latest .
# docker push your-dockerhub-username/dr-retfound-lora:latest

# Run instructions for Vast.ai:
# Mount volumes:
#   /data         -> Your datasets (APTOS, Messidor, etc.)
#   /models       -> RETFound weights
#   /results      -> Training outputs (checkpoints, logs)
#
# Example vast.ai command:
# docker run --gpus all \
#   -v /path/to/data:/data \
#   -v /path/to/models:/models \
#   -v /path/to/results:/results \
#   -it your-dockerhub-username/dr-retfound-lora:latest

# RTX 5090 Optimizations:
# - CUDA 12.1 for latest GPU architecture support
# - 32GB VRAM allows batch_size=64
# - Use mixed precision training for optimal performance
# - Expected training time: ~3-3.5 hours for 20 epochs
