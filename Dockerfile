FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="Cory Gardner"
LABEL description="Wavelet-Enhanced PaDiM for Industrial Anomaly Detection"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt /workspace/
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY data/README.md /workspace/data/README.md

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Create directories for data and results
RUN mkdir -p /workspace/data/MVTec /workspace/data/VisA /workspace/results

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
