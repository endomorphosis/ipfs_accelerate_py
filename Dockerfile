# Multi-stage Dockerfile for IPFS Accelerate Python
# Supports multi-architecture builds (amd64, arm64) with hardware acceleration
# Based on successful ipfs_kit_py patterns

ARG PYTHON_VERSION=3.11
ARG BUILD_TYPE=production
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Base stage with Python and system dependencies
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Platform information for debugging
RUN echo "Building on $BUILDPLATFORM, targeting $TARGETPLATFORM"

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

# Install package in editable mode with development dependencies
RUN pip install -e ".[all,testing,mcp,webnn,viz]"

USER appuser
ENV HOME=/home/appuser
EXPOSE 8000 5678 8888
CMD ["python", "-m", "ipfs_accelerate_py.cli_entry", "--help"]

# Testing stage
FROM development AS testing
ENV TESTING=1

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "--verbose", "--timeout=300"]

# Production build stage
FROM base AS builder

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel build

# Copy source files
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
RUN pip install --upgrade pip && \
    find /tmp -name "*.whl" -exec pip install "{}[full,mcp]" \; && \
    rm -rf /tmp/*.whl

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/models && \
    chown -R appuser:appuser /app

# Copy config files if they exist (conditional copy)
RUN mkdir -p /app/config

USER appuser
ENV HOME=/home/appuser
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import ipfs_accelerate_py; print('OK')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "ipfs_accelerate_py.cli_entry", "server", "start", "--host", "0.0.0.0", "--port", "8000"]

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

USER appuser
WORKDIR /app

CMD ["python", "-m", "ipfs_accelerate_py.cli_entry", "server", "start", "--host", "0.0.0.0", "--port", "8000", "--accelerated"]