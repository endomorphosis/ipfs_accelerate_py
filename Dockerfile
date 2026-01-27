# Multi-stage Dockerfile for IPFS Accelerate Python
# Supports multi-architecture builds (amd64, arm64) with hardware acceleration
# Based on successful ipfs_kit_py patterns

ARG PYTHON_VERSION=3.12
ARG BUILD_TYPE=production
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Base stage with Python and system dependencies
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg2 \
    software-properties-common \
    pkg-config \
    # Cryptography libraries (required for fastecdsa and other crypto packages)
    libgmp-dev \
    # Hardware acceleration libraries
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # For ARM64 hardware detection
    hwloc \
    libhwloc-dev \
    # For potential GPU support
    mesa-opencl-icd \
    ocl-icd-opencl-dev \
    # For networking and IPFS
    netcat-openbsd \
    iputils-ping \
    # For process management
    procps \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and home directory with proper ownership
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser \
    && mkdir -p /app /home/appuser/.cache /home/appuser/.local \
    && chown -R appuser:appuser /app /home/appuser \
    && chmod -R 0755 /home/appuser

# Ensure HOME points to a writable directory for pip and other tools
ENV HOME=/home/appuser \
    PIP_CACHE_DIR=/home/appuser/.cache/pip \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Development stage
FROM base AS development
ENV DEVELOPMENT=1

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    tree \
    htop \
    strace \
    gdb \
    valgrind \
    # Additional development tools for ML/AI
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Copy source code
COPY --chown=appuser:appuser . .

# Install ipfs_kit_py from GitHub known_good fork
RUN pip install --no-cache-dir git+https://github.com/endomorphosis/ipfs_kit_py.git@known_good

# Install package in editable mode with development dependencies
# Install Flask, Werkzeug, flask-cors, and fastmcp explicitly for MCP dashboard
RUN pip install flask>=3.0.0 flask-cors>=4.0.0 werkzeug>=3.0.0 fastmcp>=0.1.0 && \
    pip install -e ".[all,testing,mcp,webnn,viz]"

# Copy startup validation and entrypoint scripts
COPY --chown=appuser:appuser deployments/docker/docker_startup_check.py /app/
COPY --chown=appuser:appuser deployments/docker/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh /app/docker_startup_check.py

USER appuser
ENV HOME=/home/appuser
EXPOSE 8000 5678 8888

# Use the new entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]

# Testing stage - lightweight, only includes minimal dependencies for testing
FROM base AS testing
ENV TESTING=1

# Install Python testing tools
RUN pip install --upgrade pip setuptools wheel

# Copy source code
COPY --chown=appuser:appuser . .

# Install ipfs_kit_py from GitHub known_good fork
RUN pip install --no-cache-dir git+https://github.com/endomorphosis/ipfs_kit_py.git@known_good

# Install package with ONLY testing dependencies (not the heavy ML libs)
# This significantly reduces image size and build time
RUN pip install -e ".[minimal,testing,mcp]"

# Copy startup validation and entrypoint scripts
COPY --chown=appuser:appuser deployments/docker/docker_startup_check.py /app/
COPY --chown=appuser:appuser deployments/docker/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh /app/docker_startup_check.py

USER appuser
ENV HOME=/home/appuser

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "--verbose", "--timeout=300"]

# GPU Testing stage - includes PyTorch/CUDA for AI inference testing
# Use this stage for GPU-enabled runners (self-hosted or GPU-enabled)
FROM development AS testing-gpu
ENV TESTING=1
ENV TESTING_GPU=1

# This stage inherits all ML dependencies from development including:
# - PyTorch with CUDA support
# - Transformers
# - All AI/ML libraries needed for inference testing
# Note: This stage is larger (~6GB) and should only be used on runners
# with sufficient disk space (self-hosted runners recommended)

# Run GPU-aware tests by default
CMD ["python", "-m", "pytest", "tests/", "--verbose", "--timeout=600", "-k", "gpu or cuda or inference"]

# Production build stage
FROM base AS builder

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel build

# Copy source files - ensure all modules are included
COPY . /app/src/
WORKDIR /app/src

# Build wheel
RUN python -m build --wheel

# Production stage
FROM base AS production
ENV BUILD_TYPE=production

# Copy wheel from builder
COPY --from=builder /app/src/dist/*.whl /tmp/

# Install package with full dependencies for production
# Install Flask, Werkzeug, and flask-cors explicitly for MCP dashboard
RUN pip install --upgrade pip && \
    pip install flask>=3.0.0 flask-cors>=4.0.0 werkzeug>=3.0.0 && \
    find /tmp -name "*.whl" -exec pip install "{}[full,mcp]" \; && \
    rm -rf /tmp/*.whl

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/models && \
    chown -R appuser:appuser /app

# Copy startup validation and entrypoint scripts
COPY --chown=appuser:appuser deployments/docker/docker_startup_check.py /app/
COPY --chown=appuser:appuser deployments/docker/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh /app/docker_startup_check.py

# Copy config files if they exist (conditional copy)
RUN mkdir -p /app/config

USER appuser
ENV HOME=/home/appuser
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import ipfs_accelerate_py; print('OK')" || exit 1

EXPOSE 8000

# Use the new entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["mcp", "start", "--host", "0.0.0.0", "--port", "8000", "--dashboard", "--keep-running"]

# Minimal stage for lightweight deployments
FROM base AS minimal
ENV BUILD_TYPE=minimal

# Copy wheel from builder
COPY --from=builder /app/src/dist/*.whl /tmp/

# Install package with minimal dependencies
RUN pip install --upgrade pip && \
    find /tmp -name "*.whl" -exec pip install "{}[minimal]" \; && \
    rm -rf /tmp/*.whl

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser
WORKDIR /app

EXPOSE 8000
CMD ["python", "-m", "ipfs_accelerate_py.cli_entry", "--help"]

# Hardware-specific stage for GPU/acceleration
FROM production AS hardware-accelerated
ENV BUILD_TYPE=hardware

# Switch back to root for hardware-specific installations
USER root

# Install hardware acceleration libraries based on platform
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing AMD64 hardware acceleration libraries..."; \
        # Add NVIDIA CUDA repos if needed
        # apt-get update && apt-get install -y nvidia-cuda-toolkit || true; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Installing ARM64 hardware acceleration libraries..."; \
        # Install ARM-specific acceleration libraries
        apt-get update && apt-get install -y \
            # ARM-specific ML libraries
            libarmnn-dev \
            # Additional ARM optimization libraries
            libomp-dev \
            || true; \
    fi

# Install additional Python packages for hardware acceleration
RUN pip install \
    # Common acceleration libraries
    numpy \
    scipy \
    # Platform-specific packages will be installed if available
    && echo "Hardware acceleration setup complete"

# Copy startup validation and entrypoint scripts  
COPY --chown=appuser:appuser deployments/docker/docker_startup_check.py /app/
COPY --chown=appuser:appuser deployments/docker/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh /app/docker_startup_check.py

USER appuser
WORKDIR /app

# Use the new entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["mcp", "start", "--host", "0.0.0.0", "--port", "8000", "--dashboard", "--keep-running"]