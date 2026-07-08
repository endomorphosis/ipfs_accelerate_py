# Deployment Guides

Comprehensive guides for deploying IPFS Accelerate in various environments.

## Quick Links

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Main deployment guide
- [Cross-Platform Testing](CROSS_PLATFORM_TESTING_GUIDE.md) - Multi-platform deployment

## Deployment Options

### 1. Docker Deployment

See [Docker Guides](../docker/) for container-based deployment.

```bash
# Quick start with Docker
docker-compose up -d

# Or build from Dockerfile
docker build -t ipfs-accelerate .
docker run -p 8080:8080 ipfs-accelerate
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ipfs-accelerate
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ipfs-accelerate
  template:
    metadata:
      labels:
        app: ipfs-accelerate
    spec:
      containers:
      - name: ipfs-accelerate
        image: ipfs-accelerate:latest
        ports:
        - containerPort: 8080
```

### 3. Bare Metal Deployment

```bash
# Install dependencies
pip install ipfs-accelerate-py[full]

# Start MCP server
ipfs-accelerate mcp start --port 8080

# Start P2P node
ipfs-accelerate p2p start
```

### 4. Cloud Deployment

#### AWS

```bash
# Use ECS/EKS for container deployment
# Configure auto-scaling groups
# Setup load balancers
```

#### GCP

```bash
# Use GKE for Kubernetes deployment
# Configure Cloud Run for serverless
```

#### Azure

```bash
# Use AKS for Kubernetes deployment
# Configure Container Instances
```

## Cross-Platform Considerations

- **Linux**: Native support, optimal performance
- **macOS**: Apple Silicon (M1/M2/M3) with MPS acceleration
- **Windows**: WSL2 recommended for best compatibility
- **ARM64**: Full support on modern ARM processors

See [Cross-Platform Testing Guide](CROSS_PLATFORM_TESTING_GUIDE.md) for details.

## Production Checklist

### Security

- [ ] Configure authentication and authorization
- [ ] Enable HTTPS/TLS encryption
- [ ] Set up firewall rules
- [ ] Configure rate limiting
- [ ] Enable audit logging

### Performance

- [ ] Optimize hardware selection (GPU/CPU/TPU)
- [ ] Configure caching layers
- [ ] Set up load balancing
- [ ] Enable CDN for static assets
- [ ] Configure auto-scaling

### Monitoring

- [ ] Set up health checks
- [ ] Configure metrics collection
- [ ] Enable log aggregation
- [ ] Set up alerts and notifications
- [ ] Configure distributed tracing

### Reliability

- [ ] Configure backups
- [ ] Set up disaster recovery
- [ ] Enable high availability
- [ ] Configure failover mechanisms
- [ ] Test recovery procedures

## Environment-Specific Guides

### Development

```bash
# Local development setup
pip install -e ".[dev]"
ipfs-accelerate mcp start --debug
```

### Staging

```bash
# Staging environment with monitoring
docker-compose -f docker-compose.staging.yml up -d
```

### Production

```bash
# Production deployment with HA
kubectl apply -f k8s/production/
```

## Scaling Strategies

### Horizontal Scaling

- Add more worker nodes
- Use P2P distribution
- Configure load balancing

### Vertical Scaling

- Upgrade hardware (more CPU/GPU/RAM)
- Optimize model serving
- Use hardware acceleration

## See Also

- [Main Documentation](../../README.md)
- [Docker Guides](../docker/)
- [GitHub Guides](../github/)
- [Installation Guide](../../INSTALL.md)

---

**Last Updated**: January 2026
