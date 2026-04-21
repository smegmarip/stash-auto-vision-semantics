FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    redis-server redis-tools \
    libgl1-mesa-glx libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install PyTorch first with explicit CUDA 12.4 build (must match host driver)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install remaining Python deps (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY train/ ./train/
COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Persistent volumes
VOLUME ["/root/.cache/huggingface", "/data/redis", "/tmp/sprites"]

EXPOSE 5004

ENTRYPOINT ["./entrypoint.sh"]
