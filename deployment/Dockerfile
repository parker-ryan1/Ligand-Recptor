# Ligand-Receptor Binding Prediction - Docker Environment
# ======================================================
# Complete reproducible environment for 100% publication readiness
# Supports both CPU and GPU training with all dependencies

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
RUN pip install torch-geometric

# Install additional scientific packages
RUN pip install \
    shap \
    rdkit \
    biopandas \
    biotite \
    pytest \
    pytest-cov \
    jupyter \
    tensorboard

# Copy application code
COPY . .

# Create results directory with proper permissions
RUN mkdir -p /app/results && chmod 777 /app/results

# Create non-root user for security
RUN useradd -m -u 1000 researcher && \
    chown -R researcher:researcher /app

# Switch to non-root user
USER researcher

# Set default command
CMD ["python", "enhanced_gnn_v2_uncertainty.py"]

# Metadata
LABEL maintainer="Ligand-Receptor Research Team"
LABEL version="2.0"
LABEL description="Complete environment for ligand-receptor binding prediction with uncertainty quantification"
LABEL repository="ligand-receptor-prediction"
LABEL cuda_version="12.1"
LABEL pytorch_version="2.5.1" 