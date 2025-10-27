#!/bin/bash

# GitHub Actions Runner Setup Script for ipfs_accelerate_py
# Date: October 2025
# Description: Sets up a self-hosted GitHub Actions runner on this machine

set -euo pipefail

# Configuration
REPO_URL="https://github.com/endomorphosis/ipfs_accelerate_py"
RUNNER_VERSION="2.319.1"  # Latest stable version as of Oct 2025
RUNNER_NAME="$(hostname)-backup-runner"
RUNNER_WORK_DIR="_work"
RUNNER_LABELS="linux,x64,self-hosted,backup-runner"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check if running on Linux
    if [[ "$(uname)" != "Linux" ]]; then
        error "This script is designed for Linux systems only"
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    case "$ARCH" in
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
    
    success "System requirements check passed (Linux $ARCH)"
    
    # Check required tools
    local missing_tools=()
    
    for tool in curl tar sudo systemctl; do
        if ! command -v $tool &> /dev/null; then
            missing_tools+=($tool)
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
    
    success "All required tools are available"
}

check_token() {
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        error "GITHUB_TOKEN environment variable is required. Please set it with your GitHub PAT."
    fi
    
    if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
        warning "GITHUB_REPOSITORY not set, using default: endomorphosis/ipfs_accelerate_py"
        export GITHUB_REPOSITORY="endomorphosis/ipfs_accelerate_py"
    fi
    
    success "GitHub credentials configured"
}

create_runner_user() {
    log "Setting up runner user..."
    
    # Check if runner user already exists
    if id "actions-runner" &>/dev/null; then
        log "User 'actions-runner' already exists"
    else
        log "Creating 'actions-runner' user..."
        sudo useradd -m -s /bin/bash actions-runner
        success "Created 'actions-runner' user"
    fi
    
    # Ensure home directory exists and has correct permissions
    sudo mkdir -p /home/actions-runner
    sudo chown actions-runner:actions-runner /home/actions-runner
    sudo chmod 755 /home/actions-runner
    
    # Add to docker group if docker is installed
    if command -v docker &> /dev/null; then
        sudo usermod -aG docker actions-runner || true
        log "Added actions-runner to docker group"
    fi
    
    # Set up sudo permissions for actions-runner
    echo "actions-runner ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/actions-runner > /dev/null
    success "Configured sudo permissions for actions-runner"
}

download_and_extract_runner() {
    log "Downloading GitHub Actions runner..."
    
    local runner_dir="/home/actions-runner/actions-runner"
    
    # Create runner directory with proper permissions
    sudo mkdir -p "$runner_dir"
    sudo chown actions-runner:actions-runner "$runner_dir"
    cd "$runner_dir"
    
    # Download runner package
    local runner_file="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
    local download_url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${runner_file}"
    
    log "Downloading from: $download_url"
    sudo -u actions-runner curl -O -L "$download_url"
    
    # Verify download
    if [[ ! -f "$runner_file" ]]; then
        error "Failed to download runner package"
    fi
    
    success "Downloaded runner package"
    
    # Extract runner
    log "Extracting runner package..."
    sudo -u actions-runner tar xzf "$runner_file"
    
    # Remove archive
    sudo -u actions-runner rm "$runner_file"
    
    success "Extracted runner package"
}

get_registration_token() {
    log "Getting registration token from GitHub..."
    
    # Clean the token (remove any whitespace/newlines)
    local clean_token=$(echo "${GITHUB_TOKEN}" | tr -d '\r\n\t ')
    
    # Use GitHub API to get registration token
    local api_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners/registration-token"
    
    log "Making API request to: $api_url"
    local response=$(curl -s -X POST \
        -H "Authorization: token ${clean_token}" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "User-Agent: GitHub-Actions-Runner-Setup" \
        "$api_url")
    
    log "API Response: $response"
    
    # Check if response contains an error
    if echo "$response" | grep -q '"message"'; then
        local error_message=$(echo "$response" | grep -o '"message":"[^"]*' | cut -d'"' -f4)
        error "GitHub API Error: $error_message. Please check your token and permissions."
    fi
    
    local token=$(echo "$response" | grep -o '"token":"[^"]*' | cut -d'"' -f4)
    
    if [[ -z "$token" ]]; then
        error "Failed to get registration token. Response: $response"
    fi
    
    echo "$token"
}

configure_runner() {
    log "Configuring GitHub Actions runner..."
    
    local runner_dir="/home/actions-runner/actions-runner"
    cd "$runner_dir"
    
    # Get registration token
    local reg_token=$(get_registration_token)
    
    # Configure runner
    log "Configuring runner with name: $RUNNER_NAME"
    sudo -u actions-runner ./config.sh \
        --url "$REPO_URL" \
        --token "$reg_token" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --work "$RUNNER_WORK_DIR" \
        --unattended \
        --replace
    
    success "Runner configured successfully"
}

install_service() {
    log "Installing runner as a system service..."
    
    local runner_dir="/home/actions-runner/actions-runner"
    cd "$runner_dir"
    
    # Install service
    sudo ./svc.sh install actions-runner
    
    # Enable and start service
    sudo systemctl enable actions.runner.${GITHUB_REPOSITORY//\//.}.${RUNNER_NAME}.service
    sudo systemctl start actions.runner.${GITHUB_REPOSITORY//\//.}.${RUNNER_NAME}.service
    
    success "Runner service installed and started"
}

verify_installation() {
    log "Verifying runner installation..."
    
    local service_name="actions.runner.${GITHUB_REPOSITORY//\//.}.${RUNNER_NAME}.service"
    
    # Check service status
    if sudo systemctl is-active --quiet "$service_name"; then
        success "Runner service is active"
    else
        error "Runner service is not active"
    fi
    
    # Check if runner is online
    log "Checking runner status on GitHub..."
    sleep 5  # Give it a moment to register
    
    local api_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners"
    local runners=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" "$api_url")
    
    if echo "$runners" | grep -q "\"name\":\"$RUNNER_NAME\""; then
        success "Runner is registered and visible on GitHub"
    else
        warning "Runner may still be registering. Check GitHub Actions settings."
    fi
}

setup_project_dependencies() {
    log "Setting up project dependencies..."
    
    local project_dir="/home/actions-runner/ipfs_accelerate_py"
    
    # Clone repository if not exists
    if [[ ! -d "$project_dir" ]]; then
        log "Cloning repository..."
        sudo -u actions-runner git clone "$REPO_URL" "$project_dir"
        success "Repository cloned"
    else
        log "Repository already exists, updating..."
        cd "$project_dir"
        sudo -u actions-runner git pull
        success "Repository updated"
    fi
    
    cd "$project_dir"
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        log "Installing Python dependencies..."
        sudo -u actions-runner python3 -m pip install --user -r requirements.txt
        success "Python dependencies installed"
    fi
    
    # Install additional test dependencies if they exist
    for req_file in "test/requirements.txt" "requirements_enhanced_scraper.txt" "requirements_dashboard.txt"; do
        if [[ -f "$req_file" ]]; then
            log "Installing dependencies from $req_file..."
            sudo -u actions-runner python3 -m pip install --user -r "$req_file" || true
        fi
    done
    
    success "Project dependencies setup complete"
}

create_runner_config() {
    log "Creating runner configuration file..."
    
    local config_file="/home/actions-runner/runner-config.json"
    
    cat > "$config_file" << EOF
{
    "runner_name": "$RUNNER_NAME",
    "repository": "$GITHUB_REPOSITORY",
    "labels": "$RUNNER_LABELS",
    "work_directory": "$RUNNER_WORK_DIR",
    "installed_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "architecture": "$RUNNER_ARCH",
    "version": "$RUNNER_VERSION"
}
EOF
    
    sudo chown actions-runner:actions-runner "$config_file"
    success "Runner configuration file created"
}

show_status() {
    log "Runner installation complete! Here's the status:"
    echo
    echo "Runner Details:"
    echo "  Name: $RUNNER_NAME"
    echo "  Repository: $GITHUB_REPOSITORY"
    echo "  Labels: $RUNNER_LABELS"
    echo "  Architecture: $RUNNER_ARCH"
    echo
    echo "Service Status:"
    local service_name="actions.runner.${GITHUB_REPOSITORY//\//.}.${RUNNER_NAME}.service"
    sudo systemctl status "$service_name" --no-pager -l
    echo
    echo "To manage the runner:"
    echo "  Check status: sudo systemctl status $service_name"
    echo "  Stop runner:  sudo systemctl stop $service_name"
    echo "  Start runner: sudo systemctl start $service_name"
    echo "  View logs:    sudo journalctl -u $service_name -f"
    echo
    echo "Runner directory: /home/actions-runner/actions-runner"
    echo "Project directory: /home/actions-runner/ipfs_accelerate_py"
    echo
    success "GitHub Actions runner is ready for use!"
}

main() {
    log "Starting GitHub Actions runner setup for ipfs_accelerate_py"
    echo "Repository: $REPO_URL"
    echo "Runner Version: $RUNNER_VERSION"
    echo "Runner Name: $RUNNER_NAME"
    echo
    
    check_requirements
    check_token
    create_runner_user
    download_and_extract_runner
    configure_runner
    install_service
    setup_project_dependencies
    create_runner_config
    verify_installation
    show_status
}

# Handle command line arguments
if [[ $# -gt 0 ]]; then
    case "$1" in
        --help|-h)
            echo "GitHub Actions Runner Setup Script"
            echo
            echo "Usage: $0 [options]"
            echo
            echo "Environment Variables:"
            echo "  GITHUB_TOKEN      - GitHub Personal Access Token (required)"
            echo "  GITHUB_REPOSITORY - Repository in format owner/repo (optional, defaults to endomorphosis/ipfs_accelerate_py)"
            echo
            echo "Options:"
            echo "  --help, -h        - Show this help message"
            echo "  --status          - Show current runner status"
            echo "  --uninstall       - Uninstall the runner"
            echo
            echo "Examples:"
            echo "  export GITHUB_TOKEN='ghp_...' && $0"
            echo "  GITHUB_TOKEN='ghp_...' $0"
            exit 0
            ;;
        --status)
            local service_name="actions.runner.${GITHUB_REPOSITORY//\//.}.$(hostname)-backup-runner.service"
            sudo systemctl status "$service_name" --no-pager -l
            exit 0
            ;;
        --uninstall)
            log "Uninstalling GitHub Actions runner..."
            local runner_dir="/home/actions-runner/actions-runner"
            if [[ -d "$runner_dir" ]]; then
                cd "$runner_dir"
                sudo ./svc.sh stop || true
                sudo ./svc.sh uninstall || true
                sudo -u actions-runner ./config.sh remove --token "$(get_registration_token)" || true
            fi
            success "Runner uninstalled"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
fi

# Check if GITHUB_TOKEN is provided
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "GitHub Actions Runner Setup for ipfs_accelerate_py"
    echo
    echo "Please set your GitHub Personal Access Token:"
    echo "  export GITHUB_TOKEN='your_token_here'"
    echo "  $0"
    echo
    echo "Or run with the token:"
    echo "  GITHUB_TOKEN='your_token_here' $0"
    echo
    echo "To create a token:"
    echo "1. Go to GitHub Settings > Developer settings > Personal access tokens"
    echo "2. Create a token with 'repo' and 'admin:repo_hook' permissions"
    echo "3. Copy the token and use it with this script"
    echo
    echo "Use --help for more information."
    exit 1
fi

# Run main function
main