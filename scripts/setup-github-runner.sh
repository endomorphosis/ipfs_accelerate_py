#!/bin/bash

# GitHub Actions Runner Setup Script for Linux
# Sets up self-hosted GitHub Actions runners for the ipfs_accelerate_py project

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/endomorphosis/ipfs_accelerate_py"
RUNNER_VERSION="2.311.0"
RUNNER_NAME=""
RUNNER_LABELS="self-hosted,linux,x64,docker"
TOKEN=""
INSTALL_DEPENDENCIES=false
CREATE_SERVICE=true
RUNNER_DIR="$HOME/actions-runner"
ADDITIONAL_RUNNER=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Function to show usage
show_usage() {
    echo "GitHub Actions Runner Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --token TOKEN          GitHub Actions runner registration token (required)"
    echo "  -n, --name NAME            Runner name (default: hostname)"
    echo "  -l, --labels LABELS        Runner labels (default: self-hosted,linux,x64,docker)"
    echo "  -d, --dir DIRECTORY        Runner installation directory (default: ~/actions-runner)"
    echo "  -v, --version VERSION      Runner version (default: 2.311.0)"
    echo "  --install-deps             Install system dependencies"
    echo "  --no-service               Don't create systemd service"
    echo "  --additional               Install additional runner (for backup)"
    echo "  --verbose                  Enable verbose output"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --token XXXXXXXX --install-deps"
    echo "  $0 --token XXXXXXXX --name backup-runner --additional"
    echo "  $0 --token XXXXXXXX --labels self-hosted,linux,x64,docker,cuda"
    echo ""
    echo "Requirements:"
    echo "  - GitHub repository access token with repo permissions"
    echo "  - sudo access for service creation and dependency installation"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--token)
            TOKEN="$2"
            shift 2
            ;;
        -n|--name)
            RUNNER_NAME="$2"
            shift 2
            ;;
        -l|--labels)
            RUNNER_LABELS="$2"
            shift 2
            ;;
        -d|--dir)
            RUNNER_DIR="$2"
            shift 2
            ;;
        -v|--version)
            RUNNER_VERSION="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPENDENCIES=true
            shift
            ;;
        --no-service)
            CREATE_SERVICE=false
            shift
            ;;
        --additional)
            ADDITIONAL_RUNNER=true
            RUNNER_DIR="$HOME/actions-runner-backup"
            if [ -z "$RUNNER_NAME" ]; then
                RUNNER_NAME="$(hostname)-backup"
            fi
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate required parameters
if [ -z "$TOKEN" ]; then
    error "GitHub Actions runner token is required. Use --token option or see --help for usage."
fi

# Set default runner name if not provided
if [ -z "$RUNNER_NAME" ]; then
    if [ "$ADDITIONAL_RUNNER" = true ]; then
        RUNNER_NAME="$(hostname)-backup"
    else
        RUNNER_NAME="$(hostname)"
    fi
fi

# Detect system architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        RUNNER_ARCH="x64"
        ;;
    aarch64|arm64)
        RUNNER_ARCH="arm64"
        ;;
    *)
        error "Unsupported architecture: $ARCH"
        ;;
esac

log "Starting GitHub Actions Runner setup..."
log "Repository: $REPO_URL"
log "Runner name: $RUNNER_NAME"
log "Architecture: $RUNNER_ARCH"
log "Installation directory: $RUNNER_DIR"
log "Labels: $RUNNER_LABELS"

# Install system dependencies if requested
if [ "$INSTALL_DEPENDENCIES" = true ]; then
    log "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        log "Added user to docker group. Please log out and back in for changes to take effect."
    fi
    
    # Install Python dependencies
    if command -v python3 &> /dev/null; then
        log "Installing Python dependencies..."
        
        # Check if we're in a virtual environment
        if [[ "$VIRTUAL_ENV" != "" ]]; then
            log "Virtual environment detected: $VIRTUAL_ENV"
            python3 -m pip install --upgrade pip
            python3 -m pip install requests psutil
        else
            log "Installing to user site-packages"
            python3 -m pip install --user --upgrade pip
            python3 -m pip install --user requests psutil
        fi
    fi
fi

# Check if Docker is available and user has access
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        log "Docker is available and accessible"
        if [[ "$RUNNER_LABELS" != *"docker"* ]]; then
            RUNNER_LABELS="$RUNNER_LABELS,docker"
            log "Added 'docker' label"
        fi
    else
        warn "Docker is installed but not accessible. You may need to add user to docker group."
    fi
fi

# Check for GPU support and add appropriate labels
if command -v nvidia-smi &> /dev/null; then
    log "NVIDIA GPU detected"
    if [[ "$RUNNER_LABELS" != *"cuda"* ]]; then
        RUNNER_LABELS="$RUNNER_LABELS,cuda,gpu"
        log "Added 'cuda' and 'gpu' labels"
    fi
elif command -v rocm-smi &> /dev/null; then
    log "AMD GPU with ROCm detected"
    if [[ "$RUNNER_LABELS" != *"rocm"* ]]; then
        RUNNER_LABELS="$RUNNER_LABELS,rocm,gpu"
        log "Added 'rocm' and 'gpu' labels"
    fi
elif command -v intel_gpu_top &> /dev/null; then
    log "Intel GPU detected"
    if [[ "$RUNNER_LABELS" != *"openvino"* ]]; then
        RUNNER_LABELS="$RUNNER_LABELS,openvino,gpu"
        log "Added 'openvino' and 'gpu' labels"
    fi
else
    log "No GPU detected, using CPU-only configuration"
    if [[ "$RUNNER_LABELS" != *"cpu-only"* ]]; then
        RUNNER_LABELS="$RUNNER_LABELS,cpu-only"
        log "Added 'cpu-only' label"
    fi
fi

# Create runner directory
log "Creating runner directory: $RUNNER_DIR"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Download runner package
RUNNER_FILE="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
DOWNLOAD_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_FILE}"

if [ ! -f "$RUNNER_FILE" ]; then
    log "Downloading runner package: $RUNNER_FILE"
    curl -o "$RUNNER_FILE" -L "$DOWNLOAD_URL"
else
    log "Runner package already exists, skipping download"
fi

# Extract runner package
log "Extracting runner package..."
tar xzf "$RUNNER_FILE"

# Validate hash (optional but recommended)
if [ -f "bin/Runner.Listener" ]; then
    log "Runner package extracted successfully"
else
    error "Failed to extract runner package properly"
fi

# Configure runner
log "Configuring runner..."
log "This will prompt for configuration details..."

# Check if runner is already configured
if [ -f ".runner" ]; then
    warn "Runner appears to be already configured. Removing previous configuration..."
    if [ -f "./config.sh" ]; then
        ./config.sh remove --token "$TOKEN"
    fi
fi

# Configure the runner
./config.sh \
    --url "$REPO_URL" \
    --token "$TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work "_work" \
    --replace

log "Runner configured successfully"

# Create systemd service if requested
if [ "$CREATE_SERVICE" = true ]; then
    log "Creating systemd service..."
    
    # Install the service
    sudo ./svc.sh install "$USER"
    
    # Start the service
    sudo ./svc.sh start
    
    # Get the actual service name from the .service file
    if [ -f ".service" ]; then
        SERVICE_NAME=$(cat .service)
        log "Service created: $SERVICE_NAME"
        
        # Enable service to start on boot
        sudo systemctl enable "$SERVICE_NAME"
        
        # Check service status
        if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
            log "Service '$SERVICE_NAME' created and started successfully"
        else
            warn "Service '$SERVICE_NAME' was created but may not be running properly"
            log "Check status with: sudo systemctl status $SERVICE_NAME"
        fi
    else
        warn "Could not determine service name. Check manually with: sudo systemctl list-units | grep actions"
    fi
else
    log "Skipping service creation (--no-service specified)"
    log "To run the runner manually, use: $RUNNER_DIR/run.sh"
fi

# Create monitoring directory and logs
log "Setting up monitoring..."
sudo mkdir -p /var/log/github-actions
sudo chown $USER:$USER /var/log/github-actions

# Copy monitoring script if it exists
MONITOR_SCRIPT="$SCRIPT_DIR/scripts/monitor-runners.sh"
if [ -f "$MONITOR_SCRIPT" ]; then
    log "Setting up runner monitoring..."
    cp "$MONITOR_SCRIPT" "$RUNNER_DIR/"
    chmod +x "$RUNNER_DIR/monitor-runners.sh"
    
    # Create cron job for monitoring (optional)
    (crontab -l 2>/dev/null; echo "*/5 * * * * $RUNNER_DIR/monitor-runners.sh") | crontab -
    log "Monitoring script installed and scheduled"
fi

# Create runner health check script
cat > "$RUNNER_DIR/health-check.sh" << 'EOF'
#!/bin/bash

# Simple health check for GitHub Actions runner
SERVICE_NAME="github-actions-runner"
if [ "$1" = "backup" ]; then
    SERVICE_NAME="github-actions-runner-backup"
fi

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✓ $SERVICE_NAME is running"
    exit 0
else
    echo "✗ $SERVICE_NAME is not running"
    echo "To restart: sudo systemctl restart $SERVICE_NAME"
    exit 1
fi
EOF

chmod +x "$RUNNER_DIR/health-check.sh"

# Final status check
log "Performing final status check..."

if [ "$CREATE_SERVICE" = true ]; then
    if [ "$ADDITIONAL_RUNNER" = true ]; then
        SERVICE_NAME="github-actions-runner-backup"
    else
        SERVICE_NAME="github-actions-runner"
    fi
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "✓ Runner service is active and running"
    else
        warn "Runner service is not active. Check logs with: sudo journalctl -u $SERVICE_NAME -f"
    fi
fi

# Display summary
echo ""
log "=========================="
log "SETUP COMPLETE"
log "=========================="
log "Runner name: $RUNNER_NAME"
log "Installation directory: $RUNNER_DIR"
log "Labels: $RUNNER_LABELS"

if [ "$CREATE_SERVICE" = true ]; then
    log "Service name: $SERVICE_NAME"
    log "Service status: sudo systemctl status $SERVICE_NAME"
    log "Service logs: sudo journalctl -u $SERVICE_NAME -f"
fi

log "Health check: $RUNNER_DIR/health-check.sh"

if [ -f "$RUNNER_DIR/monitor-runners.sh" ]; then
    log "Monitoring: $RUNNER_DIR/monitor-runners.sh"
fi

echo ""
log "Useful commands:"
log "  Check runner status: $RUNNER_DIR/health-check.sh"
log "  Manual run: $RUNNER_DIR/run.sh"
log "  Remove runner: $RUNNER_DIR/config.sh remove --token YOUR_TOKEN"

if [ "$CREATE_SERVICE" = true ]; then
    log "  Restart service: sudo systemctl restart $SERVICE_NAME"
    log "  Stop service: sudo systemctl stop $SERVICE_NAME"
    log "  Start service: sudo systemctl start $SERVICE_NAME"
fi

echo ""
log "Runner setup completed successfully!"
log "The runner should now appear in your GitHub repository settings under Actions > Runners."