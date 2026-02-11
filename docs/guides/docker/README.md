# Docker Integration Guides

Comprehensive guides for Docker deployment, caching, and containerized runners.

## Quick Links

- [Docker Usage](DOCKER_USAGE.md) - Basic Docker usage guide
- [Docker Container Guide](DOCKER_CONTAINER_GUIDE.md) - Container configuration
- [Docker Cache Quick Start](DOCKER_CACHE_QUICK_START.md) - Get started with Docker cache
- [Docker Group Setup](DOCKER_GROUP_SETUP.md) - User permissions and group setup

## Cache Guides

- **[Docker Cache README](DOCKER_CACHE_README.md)** - Overview of Docker cache system
- **[Docker Cache Quick Start](DOCKER_CACHE_QUICK_START.md)** - Quick setup guide
- **[Docker Runner Cache Plan](DOCKER_RUNNER_CACHE_PLAN.md)** - Cache architecture for runners

## Container Setup

- **[Docker Usage](DOCKER_USAGE.md)** - Basic Docker commands and usage
- **[Docker Container Guide](DOCKER_CONTAINER_GUIDE.md)** - Container configuration and best practices
- **[Docker Group Setup](DOCKER_GROUP_SETUP.md)** - Configure user permissions

## Security

- **[Containerized CI Security](CONTAINERIZED_CI_SECURITY.md)** - Security best practices for containerized CI/CD

## Quick Start

### Build and Run Container

```bash
# Build the container
docker build -t ipfs-accelerate .

# Run inference container
docker run -p 8080:8080 \
  -v $(pwd)/models:/models \
  ipfs-accelerate

# Run with GPU support
docker run --gpus all \
  -p 8080:8080 \
  ipfs-accelerate
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## See Also

- [Main Documentation](../../README.md)
- [Deployment Guides](../deployment/)
- [GitHub Guides](../github/)
- [Installation Guide](../../INSTALL.md)

---

**Last Updated**: January 2026
