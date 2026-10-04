# IPFS Accelerate Docker Usage Guide

## Quick Start

### Build and Run

```bash
# Build the development image
docker build --target development -t ipfs-accelerate-py:dev .

# Run with MCP server
docker run --rm -p 9000:9000 ipfs-accelerate-py:dev mcp start

# Run with custom port
docker run --rm -p 8080:8080 -e MCP_PORT=8080 ipfs-accelerate-py:dev mcp start --port 8080

# Run with volume mounts for persistence
docker run --rm \
  -p 9000:9000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ipfs-accelerate-py:dev mcp start
```

### Using Docker Compose

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Build Targets

### Development (default)
Full development environment with all dependencies and tools.

```bash
docker build --target development -t ipfs-accelerate-py:dev .
```

**Includes:**
- All Python dependencies
- Development tools (vim, nano, htop, etc.)
- Package installed in editable mode
- Flask and Werkzeug for MCP dashboard
- All optional dependencies

### Production
Optimized production build with minimal footprint.

```bash
docker build --target production -t ipfs-accelerate-py:prod .
```

**Features:**
- Multi-stage build for smaller image size
- Package installed as wheel
- Only runtime dependencies
- Health checks enabled
- Non-root user (appuser)

### Testing
Pre-configured for running tests.

```bash
docker build --target testing -t ipfs-accelerate-py:test .
docker run --rm ipfs-accelerate-py:test
```

### Minimal
Lightweight image with core functionality only.

```bash
docker build --target minimal -t ipfs-accelerate-py:minimal .
```

### Hardware-Accelerated
Includes drivers and libraries for GPU acceleration.

```bash
docker build --target hardware-accelerated -t ipfs-accelerate-py:gpu .

# Run with NVIDIA GPU
docker run --rm --gpus all -p 9000:9000 ipfs-accelerate-py:gpu mcp start
```

## Multi-Architecture Builds

### Build for AMD64 and ARM64

```bash
# Create buildx builder
docker buildx create --name multiarch --use

# Build for both architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  -t your-registry/ipfs-accelerate-py:latest \
  --push \
  .
```

### Architecture-Specific Features

The container automatically detects the architecture and:
- ✅ Filters incompatible workflow requirements
- ✅ Installs architecture-appropriate packages
- ✅ Optimizes performance for the platform

## Startup Validation

Every container startup performs comprehensive validation:

### System Checks
- ✅ Architecture compatibility (x86_64, arm64)
- ✅ Python environment (version, packages)
- ✅ Container environment detection

### Dependency Checks  
- ✅ Core Python packages (pip, setuptools, wheel)
- ✅ IPFS Accelerate package import
- ✅ Required modules (cli, mcp, shared)
- ✅ System dependencies (curl, wget, git)

### Infrastructure Checks
- ✅ Hardware acceleration (CUDA, ROCm, OpenCL)
- ✅ Network connectivity (DNS, HTTPS)
- ✅ File system permissions
- ✅ Port availability

### MCP Server Checks
- ✅ Flask and Werkzeug dependencies
- ✅ MCP module availability
- ✅ Port binding capability

## Environment Variables

### MCP Server Configuration

```bash
MCP_HOST=0.0.0.0           # Server bind address
MCP_PORT=9000              # Server port
MCP_DASHBOARD=true         # Enable dashboard
MCP_KEEP_RUNNING=true      # Keep server running
```

### Python Configuration

```bash
PYTHONUNBUFFERED=1         # Unbuffered output
PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files
```

### Paths

```bash
HOME=/home/appuser         # User home directory
WORKDIR=/app               # Application directory
```

## Volume Mounts

### Recommended Volumes

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \           # Persistent data
  -v $(pwd)/models:/app/models \       # Model cache
  -v $(pwd)/logs:/app/logs \           # Application logs
  -v $(pwd)/config:/app/config \       # Configuration files
  ipfs-accelerate-py:dev mcp start
```

### Read-Only Mounts

```bash
docker run --rm \
  -v $(pwd)/config:/app/config:ro \    # Read-only config
  ipfs-accelerate-py:dev mcp start
```

## Networking

### Port Mappings

```bash
-p 9000:9000   # MCP server and dashboard
-p 8000:8000   # Alternative MCP port
-p 5000:5000   # Flask development server
```

### Custom Network

```bash
# Create network
docker network create ipfs-network

# Run container
docker run --rm \
  --network ipfs-network \
  --name ipfs-accelerate \
  -p 9000:9000 \
  ipfs-accelerate-py:dev mcp start
```

## Health Checks

Production images include health checks:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' <container-id>

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' <container-id>
```

## Troubleshooting

### View Startup Validation

```bash
docker run --rm ipfs-accelerate-py:dev --help
```

The validation output shows:
- System information
- Dependency status
- Configuration issues
- Recommendations

### Debug Mode

```bash
# Enable verbose validation
docker run --rm -e VALIDATION_VERBOSE=1 ipfs-accelerate-py:dev mcp start

# Access container shell
docker run --rm -it ipfs-accelerate-py:dev /bin/bash

# Run validation manually
docker run --rm -it ipfs-accelerate-py:dev /bin/bash
python /app/docker_startup_check.py --verbose
```

### Check Logs

```bash
# Follow container logs
docker logs -f <container-id>

# View last 100 lines
docker logs --tail 100 <container-id>

# Filter for errors
docker logs <container-id> 2>&1 | grep -i error
```

### Common Issues

#### Port Already in Use

```bash
# Check what's using the port
lsof -i :9000

# Use different port
docker run --rm -p 9001:9001 ipfs-accelerate-py:dev mcp start --port 9001
```

#### Permission Denied

```bash
# Run with user mapping
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data \
  ipfs-accelerate-py:dev mcp start
```

#### Missing Dependencies

```bash
# Rebuild with --no-cache
docker build --no-cache --target development -t ipfs-accelerate-py:dev .
```

## Performance Optimization

### Resource Limits

```bash
# Limit CPU and memory
docker run --rm \
  --cpus=4 \
  --memory=8g \
  -p 9000:9000 \
  ipfs-accelerate-py:dev mcp start
```

### Build Cache

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t ipfs-accelerate-py:dev .

# Clear build cache if needed
docker builder prune -af
```

## Security

### Non-Root User

All images run as `appuser` (UID 1000) by default.

### Read-Only Root Filesystem

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/logs \
  -p 9000:9000 \
  ipfs-accelerate-py:prod mcp start
```

### Security Scanning

```bash
# Scan image for vulnerabilities
docker scan ipfs-accelerate-py:prod
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build and test Docker image
  run: |
    docker build --target testing -t ipfs-accelerate-py:test .
    docker run --rm ipfs-accelerate-py:test
```

### GitLab CI

```yaml
test:docker:
  script:
    - docker build --target testing -t $CI_REGISTRY_IMAGE:test .
    - docker run --rm $CI_REGISTRY_IMAGE:test
```

## Examples

### Complete Development Setup

```bash
# Clone repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Build development image
docker build --target development -t ipfs-accelerate-py:dev .

# Run with all features
docker run --rm -it \
  -p 9000:9000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name ipfs-accelerate-dev \
  ipfs-accelerate-py:dev mcp start

# Access in browser
open http://localhost:9000/dashboard
```

### Production Deployment

```bash
# Build production image
docker build --target production -t ipfs-accelerate-py:latest .

# Run in production
docker run -d \
  --name ipfs-accelerate-prod \
  --restart unless-stopped \
  -p 9000:9000 \
  -v /data/ipfs:/app/data \
  -v /data/models:/app/models \
  -v /var/log/ipfs:/app/logs \
  --health-cmd="python -c 'import ipfs_accelerate_py; print(\"OK\")'" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  ipfs-accelerate-py:latest mcp start
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Documentation: https://github.com/endomorphosis/ipfs_accelerate_py/blob/main/README.md
