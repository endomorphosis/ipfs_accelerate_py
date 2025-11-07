# IPFS Accelerate Docker Container Guide

## Overview

The IPFS Accelerate Docker container includes comprehensive dependency validation and system checks that run at startup to ensure proper operation across different architectures and operating systems.

## Features

✅ **Automatic Dependency Validation** - Checks all dependencies at container startup
✅ **Multi-Architecture Support** - Works on x86_64 (amd64) and ARM64 (aarch64) 
✅ **Hardware Acceleration Detection** - Automatically detects CUDA, ROCm, OpenCL
✅ **System Health Checks** - Validates Python environment, packages, permissions
✅ **Flexible Entrypoint** - Easy command execution with validation

## Quick Start

### Build the Container

```bash
# Production build
docker build -t ipfs-accelerate-py:latest .

# Development build
docker build --target development -t ipfs-accelerate-py:dev .

# Hardware-accelerated build
docker build --target hardware-accelerated -t ipfs-accelerate-py:hw .
```

### Run the MCP Server

```bash
# Start MCP server with dashboard
docker run -p 8000:8000 ipfs-accelerate-py:latest mcp start

# Start with custom port
docker run -p 9000:9000 ipfs-accelerate-py:latest mcp start --port 9000

# Start with volume mounts for persistence
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  ipfs-accelerate-py:latest mcp start
```

### Run Other Commands

```bash
# Show help
docker run --rm ipfs-accelerate-py:latest --help

# Run validation only
docker run --rm ipfs-accelerate-py:latest validate

# Check MCP server status
docker run --rm ipfs-accelerate-py:latest mcp status

# Interactive shell
docker run -it ipfs-accelerate-py:latest bash
```

## Container Startup Process

When the container starts, it automatically:

1. **Displays Environment Info** - Shows architecture, OS, Python version
2. **Runs Dependency Validation** - Checks all system and Python dependencies
3. **Validates System Configuration** - Ensures proper permissions and setup
4. **Detects Hardware** - Identifies available acceleration (GPU, etc.)
5. **Starts Requested Service** - Runs your command if validation passes

### Validation Checks

The startup script (`docker_startup_check.py`) performs these checks:

- ✅ **System Information** - Architecture, OS, CPU, memory
- ✅ **Python Environment** - Version, pip, setuptools, wheel
- ✅ **Package Installation** - ipfs_accelerate_py and dependencies
- ✅ **System Dependencies** - curl, wget, git, etc.
- ✅ **Hardware Acceleration** - CUDA, ROCm, OpenCL detection
- ✅ **Network Connectivity** - DNS resolution, HTTPS access
- ✅ **File Permissions** - Write access to required directories
- ✅ **MCP Requirements** - Flask and related dependencies

## Multi-Architecture Support

The container automatically detects and configures for different architectures:

### x86_64 (AMD64)
```bash
docker build --platform linux/amd64 -t ipfs-accelerate-py:amd64 .
docker run --platform linux/amd64 ipfs-accelerate-py:amd64 mcp start
```

### ARM64 (AArch64)
```bash
docker build --platform linux/arm64 -t ipfs-accelerate-py:arm64 .
docker run --platform linux/arm64 ipfs-accelerate-py:arm64 mcp start
```

### Multi-platform Build
```bash
# Build for both architectures
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ipfs-accelerate-py:multiarch .
```

## Docker Compose

Use Docker Compose for easier management:

```bash
# Start production service
docker-compose up ipfs-accelerate-py

# Start development service
docker-compose up ipfs-accelerate-py-dev

# View logs
docker-compose logs -f ipfs-accelerate-py

# Stop services
docker-compose down
```

## Testing the Container

Run the comprehensive test suite:

```bash
# Run all tests
./test-docker-container.sh

# Test specific target
BUILD_TARGET=development ./test-docker-container.sh

# Test with custom image name
IMAGE_NAME=my-custom-image:test ./test-docker-container.sh
```

The test suite validates:
- Container startup
- Dependency validation
- Package imports
- CLI commands
- System information
- Architecture detection
- Network connectivity
- File permissions
- MCP server startup
- Multi-architecture compatibility

## Environment Variables

Configure the container with environment variables:

```bash
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  -e PYTHONUNBUFFERED=1 \
  -e IPFS_ACCELERATE_MODE=production \
  ipfs-accelerate-py:latest mcp start
```

Available variables:
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONUNBUFFERED` - Unbuffered Python output (1 or 0)
- `PYTHONPATH` - Additional Python paths
- `IPFS_ACCELERATE_MODE` - Operation mode (production, development, test)

## Volume Mounts

Persist data and configurations with volumes:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \           # Data directory
  -v $(pwd)/logs:/app/logs \           # Log files
  -v $(pwd)/config:/app/config:ro \    # Configuration (read-only)
  -v $(pwd)/models:/app/models \       # Model storage
  ipfs-accelerate-py:latest mcp start
```

## Troubleshooting

### Container Fails Validation

If validation fails, check the startup logs:

```bash
docker run --rm ipfs-accelerate-py:latest validate --verbose
```

### Import Errors

Ensure PYTHONPATH is set correctly:

```bash
docker run --rm -e PYTHONPATH=/app ipfs-accelerate-py:latest python3 -c "import ipfs_accelerate_py; print('OK')"
```

### Permission Issues

Run as non-root user (default: appuser):

```bash
docker run --user appuser ipfs-accelerate-py:latest mcp start
```

### Network Issues

Test network connectivity:

```bash
docker run --rm ipfs-accelerate-py:latest python3 -c "import urllib.request; urllib.request.urlopen('https://google.com'); print('Network OK')"
```

### Debug Mode

Run with verbose validation and debug logging:

```bash
docker run --rm \
  -e LOG_LEVEL=DEBUG \
  ipfs-accelerate-py:latest validate --verbose
```

## Architecture-Specific Validation

The container automatically validates architecture-specific features:

### AMD64/x86_64
- Checks for NVIDIA CUDA support
- Validates x86-specific optimizations
- Tests SSE/AVX instructions

### ARM64/AArch64  
- Checks for ARM NN libraries
- Validates NEON optimizations
- Tests ARM-specific acceleration

## Health Checks

The container includes a health check that runs every 30 seconds:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' <container-id>

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' <container-id>
```

## Production Deployment

### With Docker Compose

```yaml
services:
  ipfs-accelerate:
    image: ipfs-accelerate-py:latest
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - IPFS_ACCELERATE_MODE=production
    healthcheck:
      test: ["CMD", "python3", "-m", "ipfs_accelerate_py.cli_entry", "mcp", "status"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### With Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ipfs-accelerate
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: ipfs-accelerate
        image: ipfs-accelerate-py:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
```

## Advanced Usage

### Custom Validation

Skip validation (not recommended):

```bash
# Override entrypoint to skip validation
docker run --rm --entrypoint python3 ipfs-accelerate-py:latest -m ipfs_accelerate_py.cli_entry mcp start
```

### Development Mode

Run with live code mounting:

```bash
docker run -it \
  -v $(pwd):/app \
  -p 8000:8000 \
  ipfs-accelerate-py:dev \
  bash
```

### GPU Support

For NVIDIA GPU support:

```bash
docker run --gpus all \
  -p 8000:8000 \
  ipfs-accelerate-py:hw \
  mcp start
```

## Summary

The IPFS Accelerate Docker container provides:

- ✅ **Automatic validation** of all dependencies at startup
- ✅ **Multi-architecture support** for AMD64 and ARM64
- ✅ **Hardware detection** for GPUs and accelerators
- ✅ **Comprehensive checks** for system configuration
- ✅ **Flexible commands** via smart entrypoint
- ✅ **Production-ready** deployment options

All validation happens automatically when you run:
```bash
docker run -p 8000:8000 ipfs-accelerate-py:latest mcp start
```

The container will validate everything and provide detailed feedback before starting the MCP server.
