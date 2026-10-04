#!/bin/bash
# Advanced Production Deployment Script
# Comprehensive enterprise deployment automation for IPFS Accelerate Python

set -euo pipefail

# Configuration
DEPLOYMENT_ENV="${1:-production}"
DEPLOYMENT_TARGET="${2:-kubernetes}"
REPLICAS="${3:-3}"
ENABLE_MONITORING="${4:-true}"
ENABLE_SSL="${5:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Pre-deployment validation
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
    command -v python3 >/dev/null 2>&1 || error "Python 3 is required but not installed"
    
    if [[ "$DEPLOYMENT_TARGET" == "kubernetes" ]]; then
        command -v kubectl >/dev/null 2>&1 || error "kubectl is required for Kubernetes deployment"
        kubectl cluster-info >/dev/null 2>&1 || error "Kubernetes cluster not accessible"
    fi
    
    # Validate environment
    if [[ ! "$DEPLOYMENT_ENV" =~ ^(development|staging|production)$ ]]; then
        error "Invalid environment. Must be: development, staging, or production"
    fi
    
    success "Environment validation completed"
}

# Build and tag Docker image
build_image() {
    log "Building Docker image for $DEPLOYMENT_ENV environment..."
    
    local IMAGE_TAG="ipfs-accelerate-py:$DEPLOYMENT_ENV-$(date +%Y%m%d-%H%M%S)"
    
    docker build \
        --tag "$IMAGE_TAG" \
        --tag "ipfs-accelerate-py:$DEPLOYMENT_ENV-latest" \
        --build-arg ENVIRONMENT="$DEPLOYMENT_ENV" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    success "Docker image built: $IMAGE_TAG"
    echo "$IMAGE_TAG" > .last_image_tag
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes cluster..."
    
    # Apply namespace
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ipfs-accelerate-$DEPLOYMENT_ENV
  labels:
    environment: $DEPLOYMENT_ENV
    application: ipfs-accelerate-py
EOF

    # Create ConfigMap
    kubectl create configmap ipfs-accelerate-config \
        --from-literal=ENVIRONMENT="$DEPLOYMENT_ENV" \
        --from-literal=REPLICAS="$REPLICAS" \
        --from-literal=MONITORING_ENABLED="$ENABLE_MONITORING" \
        --namespace="ipfs-accelerate-$DEPLOYMENT_ENV" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply deployment
    envsubst < kubernetes.yaml | kubectl apply -f - --namespace="ipfs-accelerate-$DEPLOYMENT_ENV"
    
    # Wait for rollout
    kubectl rollout status deployment/ipfs-accelerate-deployment \
        --namespace="ipfs-accelerate-$DEPLOYMENT_ENV" \
        --timeout=300s
    
    success "Kubernetes deployment completed"
}

# Deploy to Docker Swarm
deploy_docker_swarm() {
    log "Deploying to Docker Swarm..."
    
    # Initialize swarm if not already initialized
    if ! docker info | grep -q "Swarm: active"; then
        docker swarm init
    fi
    
    # Deploy stack
    ENVIRONMENT="$DEPLOYMENT_ENV" \
    REPLICAS="$REPLICAS" \
    MONITORING_ENABLED="$ENABLE_MONITORING" \
    docker stack deploy -c docker-compose.yml ipfs-accelerate
    
    success "Docker Swarm deployment completed"
}

# Deploy locally with Docker Compose
deploy_local() {
    log "Deploying locally with Docker Compose..."
    
    ENVIRONMENT="$DEPLOYMENT_ENV" \
    REPLICAS="$REPLICAS" \
    MONITORING_ENABLED="$ENABLE_MONITORING" \
    docker-compose up -d --scale app="$REPLICAS"
    
    success "Local deployment completed"
}

# Setup monitoring
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "Setting up monitoring stack..."
        
        case "$DEPLOYMENT_TARGET" in
            kubernetes)
                kubectl apply -f monitoring/ --namespace="ipfs-accelerate-$DEPLOYMENT_ENV"
                ;;
            docker|local)
                docker-compose -f monitoring/docker-compose.monitoring.yml up -d
                ;;
        esac
        
        success "Monitoring stack deployed"
    else
        warning "Monitoring disabled"
    fi
}

# Setup SSL/TLS
setup_ssl() {
    if [[ "$ENABLE_SSL" == "true" ]]; then
        log "Setting up SSL/TLS certificates..."
        
        # Generate self-signed certificates for development
        if [[ "$DEPLOYMENT_ENV" == "development" ]]; then
            mkdir -p ssl/
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout ssl/tls.key \
                -out ssl/tls.crt \
                -subj "/C=US/ST=CA/L=SF/O=IPFS/CN=localhost"
        fi
        
        success "SSL/TLS setup completed"
    else
        warning "SSL/TLS disabled"
    fi
}

# Health check and validation
validate_deployment() {
    log "Validating deployment health..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        case "$DEPLOYMENT_TARGET" in
            kubernetes)
                local ready_replicas=$(kubectl get deployment ipfs-accelerate-deployment \
                    --namespace="ipfs-accelerate-$DEPLOYMENT_ENV" \
                    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
                
                if [[ "$ready_replicas" -ge "$REPLICAS" ]]; then
                    success "All $REPLICAS replicas are ready"
                    break
                fi
                ;;
            docker|local)
                local running_containers=$(docker ps --filter "label=com.docker.compose.service=app" --format "{{.Names}}" | wc -l)
                
                if [[ "$running_containers" -ge "$REPLICAS" ]]; then
                    success "All $REPLICAS containers are running"
                    break
                fi
                ;;
        esac
        
        log "Waiting for deployment to be ready... ($attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        error "Deployment validation failed after $max_attempts attempts"
    fi
}

# Cleanup on failure
cleanup_on_failure() {
    error "Deployment failed, cleaning up..."
    
    case "$DEPLOYMENT_TARGET" in
        kubernetes)
            kubectl delete namespace "ipfs-accelerate-$DEPLOYMENT_ENV" --ignore-not-found=true
            ;;
        docker)
            docker stack rm ipfs-accelerate
            ;;
        local)
            docker-compose down
            ;;
    esac
}

# Main deployment function
main() {
    log "Starting advanced production deployment..."
    log "Environment: $DEPLOYMENT_ENV"
    log "Target: $DEPLOYMENT_TARGET"
    log "Replicas: $REPLICAS"
    log "Monitoring: $ENABLE_MONITORING"
    log "SSL: $ENABLE_SSL"
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Execute deployment steps
    validate_environment
    build_image
    setup_ssl
    
    case "$DEPLOYMENT_TARGET" in
        kubernetes)
            deploy_kubernetes
            ;;
        docker)
            deploy_docker_swarm
            ;;
        local)
            deploy_local
            ;;
        *)
            error "Unsupported deployment target: $DEPLOYMENT_TARGET"
            ;;
    esac
    
    setup_monitoring
    validate_deployment
    
    success "🎉 Advanced production deployment completed successfully!"
    
    # Display access information
    log "Deployment Information:"
    case "$DEPLOYMENT_TARGET" in
        kubernetes)
            echo "Namespace: ipfs-accelerate-$DEPLOYMENT_ENV"
            echo "Service: kubectl get service --namespace=ipfs-accelerate-$DEPLOYMENT_ENV"
            ;;
        docker|local)
            echo "Services: docker-compose ps"
            echo "Logs: docker-compose logs"
            ;;
    esac
    
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "Monitoring available at:"
        echo "- Prometheus: http://localhost:9090"
        echo "- Grafana: http://localhost:3000"
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi