# IPFS Accelerate Py - CI/CD Quick Reference Guide

## 🎯 Overview

This repository now has comprehensive multi-architecture CI/CD support with three main workflows:

1. **Enhanced CI/CD** (`enhanced-ci-cd.yml`) - Original comprehensive workflow
2. **AMD64 CI** (`amd64-ci.yml`) - AMD64-specific optimized testing
3. **Multi-Arch CI** (`multiarch-ci.yml`) - Comprehensive cross-platform validation

## 🚀 Workflows Summary

### Enhanced CI/CD Pipeline
- **Trigger:** Push to main/develop, PRs, manual dispatch
- **Platforms:** AMD64 + ARM64 (conditional)
- **Features:** System detection, Docker builds, Compose testing
- **Duration:** ~15-25 minutes

### AMD64 CI Pipeline  
- **Trigger:** Push to main/develop, PRs, daily schedule
- **Platforms:** AMD64 (x86_64) only
- **Features:** Native + Docker + Performance + Integration testing
- **Duration:** ~10-15 minutes

### Multi-Architecture CI Pipeline
- **Trigger:** Push to main/develop, PRs, daily schedule, manual dispatch
- **Platforms:** AMD64 + ARM64 (configurable)
- **Features:** Full cross-platform validation, security scanning, benchmarks
- **Duration:** ~25-40 minutes

## 🎮 Manual Workflow Controls

### Enhanced CI/CD Workflow
```yaml
# Manual dispatch options:
test_hardware_acceleration: true/false  # Test GPU support
run_comprehensive_tests: true/false     # Extended test suite  
```

### AMD64 CI Workflow
```yaml
# Manual dispatch options:
build_gpu_images: true/false            # Build GPU-accelerated images
```

### Multi-Architecture CI Workflow
```yaml
# Manual dispatch options:
test_all_platforms: true/false          # Test both AMD64 and ARM64
build_gpu_images: true/false            # Build GPU-accelerated images  
performance_testing: true/false         # Run performance benchmarks
```

## 🐳 Docker Targets Available

### Multi-Stage Build Targets

| Target | Purpose | Size | Use Case |
|--------|---------|------|----------|
| `base` | Foundation layer | Small | Base for other targets |
| `development` | Dev environment | Large | Local development |
| `testing` | Testing setup | Medium | CI/CD testing |
| `production` | Production ready | Medium | Deployment |
| `minimal` | Lightweight | Smallest | Resource-constrained |
| `hardware-accelerated` | GPU support | Large | ML/AI workloads |

### Building Specific Targets

```bash
# Build minimal image for ARM64
docker buildx build --platform linux/arm64 --target minimal -t ipfs-accelerate-py:minimal-arm64 .

# Build GPU image for AMD64  
docker buildx build --platform linux/amd64 --target hardware-accelerated -t ipfs-accelerate-py:gpu-amd64 .

# Build multi-arch production image
docker buildx build --platform linux/amd64,linux/arm64 --target production -t ipfs-accelerate-py:production .
```

## 🎛️ Docker Compose Profiles

### Available Profiles

| Profile | Services | Purpose |
|---------|----------|---------|
| `minimal` | Main app only | Basic functionality |
| `development` | App + dev tools | Development environment |
| `production` | App + monitoring + IPFS | Production deployment |
| `analytics` | + DuckDB + Jupyter | Data analysis |
| `gpu` | + GPU acceleration | Hardware acceleration |

### Using Profiles

```bash
# Start minimal setup
docker compose --profile minimal up -d

# Start development environment  
docker compose --profile development up -d

# Start full production stack
docker compose --profile production up -d

# Start with analytics capabilities
docker compose --profile production --profile analytics up -d

# Start with GPU support
docker compose --profile production --profile gpu up -d
```

## 🔧 Local Testing Commands

### Quick Docker Tests

```bash
# Test all Docker targets locally
python test_docker_multiarch.py

# Test specific platform
docker buildx build --platform linux/arm64 --target minimal -t test:arm64 .
docker run --platform linux/arm64 test:arm64 python -c "import ipfs_accelerate_py; print('✅ ARM64 OK')"
```

### Manual Cross-Platform Testing

```bash
# Setup buildx for cross-platform builds
docker buildx create --use --name multiarch-builder

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 --target production -t ipfs-accelerate-py:multiarch .

# Test ARM64 emulation locally  
docker run --platform linux/arm64 --rm ipfs-accelerate-py:multiarch python --version
```

## 📊 CI/CD Test Matrix

### Native Testing Matrix
- **Python Versions:** 3.9, 3.10, 3.11, 3.12  
- **Platforms:** AMD64 (always), ARM64 (conditional)
- **OS:** Ubuntu Latest

### Docker Testing Matrix  
- **Targets:** minimal, development, production
- **Platforms:** linux/amd64, linux/arm64
- **Caching:** GitHub Actions cache optimization

### Performance Testing
- **Metrics:** Import time, memory usage, CPU performance
- **Platforms:** Cross-platform comparison
- **Benchmarks:** String operations, basic computations

## 🛡️ Security Features

### Container Scanning
- **Tool:** Trivy vulnerability scanner
- **Target:** Production images
- **Output:** SARIF format for GitHub Security tab

### Best Practices
- Non-root user in containers
- Minimal attack surface
- Multi-stage builds for size optimization
- Security-focused base images

## 📈 Monitoring & Artifacts

### Workflow Artifacts
- Performance benchmarks (30 days retention)
- Test summaries (30 days retention)  
- Docker build logs
- Security scan results

### Health Checks
- Container health endpoints
- Service dependency checks
- Resource usage monitoring

## 🚨 Troubleshooting

### Common Issues

**ARM64 Build Slow/Timeout:**
```bash
# Use local ARM64 runner or increase timeout
timeout: 1800  # 30 minutes for ARM64 builds
```

**Docker Build Context Too Large:**
```bash
# Add .dockerignore file
echo "*.md\ntests/\n.git/\n__pycache__/" > .dockerignore
```

**Cross-Platform Dependencies:**  
```bash
# Platform-specific package installation
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install arm64-optimized-package; \
    fi
```

### Debug Commands

```bash
# Check buildx platforms
docker buildx ls

# Inspect multi-arch image  
docker buildx imagetools inspect ipfs-accelerate-py:multiarch

# Test platform emulation
docker run --platform linux/arm64 --rm alpine uname -m
```

## 🎯 Performance Optimization Tips

### Build Speed
- Use multi-stage builds
- Leverage build cache
- Minimize context size
- Parallel builds with buildx

### Runtime Performance  
- Choose appropriate base images
- Optimize for target architecture
- Use hardware acceleration when available
- Profile memory and CPU usage

## 📝 Workflow Status Badges

Add to your README.md:

```markdown
[![Enhanced CI/CD](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/enhanced-ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/enhanced-ci-cd.yml)

[![AMD64 CI](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/amd64-ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/amd64-ci.yml)

[![Multi-Arch CI](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/multiarch-ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ipfs_accelerate_py/actions/workflows/multiarch-ci.yml)
```

## 🔄 Continuous Integration Strategy

### Branch Protection
- Require status checks from critical workflows
- Require up-to-date branches
- Restrict pushes to main branch

### Workflow Scheduling
- **Daily:** Multi-architecture validation  
- **On Push:** AMD64 fast feedback
- **On PR:** Full enhanced CI/CD suite
- **Manual:** GPU and performance testing

---

*Generated for IPFS Accelerate Py CI/CD Pipeline*  
*Last Updated: $(date)*