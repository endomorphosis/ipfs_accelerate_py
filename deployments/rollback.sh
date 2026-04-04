#!/bin/bash
# Automated rollback script for IPFS Accelerate Python

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/tmp/ipfs_accelerate_backup}"
ROLLBACK_VERSION="${1:-previous}"
SERVICE_NAME="ipfs-accelerate-py"
NAMESPACE="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running in Kubernetes
if command -v kubectl &> /dev/null; then
    DEPLOYMENT_TYPE="kubernetes"
elif command -v docker-compose &> /dev/null; then
    DEPLOYMENT_TYPE="docker-compose"
elif command -v docker &> /dev/null; then
    DEPLOYMENT_TYPE="docker"
else
    error "No supported deployment type found (kubectl, docker-compose, or docker)"
    exit 1
fi

log "Starting automated rollback for $SERVICE_NAME (type: $DEPLOYMENT_TYPE)"

perform_kubernetes_rollback() {
    log "Performing Kubernetes rollback..."
    
    # Check if deployment exists
    if ! kubectl get deployment $SERVICE_NAME -n $NAMESPACE &>/dev/null; then
        error "Deployment $SERVICE_NAME not found in namespace $NAMESPACE"
        exit 1
    fi
    
    # Get current revision
    CURRENT_REVISION=$(kubectl rollout history deployment/$SERVICE_NAME -n $NAMESPACE --revision=0 | tail -1 | awk '{print $1}')
    log "Current revision: $CURRENT_REVISION"
    
    # Rollback to previous revision
    log "Rolling back deployment..."
    kubectl rollout undo deployment/$SERVICE_NAME -n $NAMESPACE
    
    # Wait for rollback to complete
    log "Waiting for rollback to complete..."
    kubectl rollout status deployment/$SERVICE_NAME -n $NAMESPACE --timeout=300s
    
    # Verify rollback
    NEW_REVISION=$(kubectl rollout history deployment/$SERVICE_NAME -n $NAMESPACE --revision=0 | tail -1 | awk '{print $1}')
    if [[ "$NEW_REVISION" != "$CURRENT_REVISION" ]]; then
        log "Rollback successful! New revision: $NEW_REVISION"
    else
        error "Rollback may have failed - revision unchanged"
        exit 1
    fi
}

perform_docker_compose_rollback() {
    log "Performing Docker Compose rollback..."
    
    if [[ ! -f "docker-compose.yml" ]]; then
        error "docker-compose.yml not found in current directory"
        exit 1
    fi
    
    # Stop current services
    log "Stopping current services..."
    docker-compose down
    
    # Restore from backup if available
    if [[ -d "$BACKUP_DIR" ]]; then
        log "Restoring from backup..."
        cp -r "$BACKUP_DIR"/* .
    else
        warn "No backup directory found at $BACKUP_DIR"
    fi
    
    # Start services with previous configuration
    log "Starting services with previous configuration..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check health
    if docker-compose ps | grep -q "Up"; then
        log "Rollback successful! Services are running"
    else
        error "Rollback failed - services are not running properly"
        exit 1
    fi
}

perform_docker_rollback() {
    log "Performing Docker rollback..."
    
    # Get current container
    CURRENT_CONTAINER=$(docker ps -q --filter "name=$SERVICE_NAME")
    
    if [[ -z "$CURRENT_CONTAINER" ]]; then
        warn "No running container found with name $SERVICE_NAME"
    else
        log "Stopping current container..."
        docker stop $CURRENT_CONTAINER || true
        docker rm $CURRENT_CONTAINER || true
    fi
    
    # Find previous image
    PREVIOUS_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | grep $SERVICE_NAME | sed -n '2p' | awk '{print $1}')
    
    if [[ -z "$PREVIOUS_IMAGE" ]]; then
        error "No previous image found for $SERVICE_NAME"
        exit 1
    fi
    
    log "Rolling back to image: $PREVIOUS_IMAGE"
    
    # Start container with previous image
    docker run -d --name $SERVICE_NAME --restart unless-stopped -p 8000:8000 $PREVIOUS_IMAGE
    
    # Wait and check health
    sleep 30
    if docker ps | grep -q $SERVICE_NAME; then
        log "Rollback successful! Container is running"
    else
        error "Rollback failed - container is not running"
        exit 1
    fi
}

# Perform health check after rollback
perform_health_check() {
    log "Performing post-rollback health check..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Try to connect to health endpoint
    for i in {1..5}; do
        if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
            log "Health check passed!"
            return 0
        fi
        warn "Health check attempt $i/5 failed, retrying in 10s..."
        sleep 10
    done
    
    error "Health check failed after rollback!"
    return 1
}

# Main rollback logic
case $DEPLOYMENT_TYPE in
    kubernetes)
        perform_kubernetes_rollback
        ;;
    docker-compose)
        perform_docker_compose_rollback
        ;;
    docker)
        perform_docker_rollback
        ;;
esac

# Perform health check
if perform_health_check; then
    log "Automated rollback completed successfully!"
    exit 0
else
    error "Rollback completed but health check failed!"
    exit 1
fi