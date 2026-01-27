#!/usr/bin/env bash
#
# Zero-Touch Installer for ipfs_accelerate_py Cache Infrastructure
# Supports: Linux (x86_64, ARM64, ARMv7), macOS (Intel, Apple Silicon), FreeBSD
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/installers/install.sh | bash
#   or
#   ./install.sh [--profile minimal|standard|full|cli] [--silent] [--no-cli-tools]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_PROFILE="${INSTALL_PROFILE:-standard}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/ipfs_accelerate_py}"
SILENT=false
NO_CLI_TOOLS=false
SKIP_VENV=false
VERIFY_ONLY=false
VERBOSE=false

# Logging
LOG_FILE="$HOME/.cache/ipfs_accelerate_py/install.log"

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Write to log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Print to console if not silent
    if [ "$SILENT" = false ]; then
        case $level in
            INFO)
                echo -e "${BLUE}ℹ${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}✓${NC} $message"
                ;;
            WARN)
                echo -e "${YELLOW}⚠${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}✗${NC} $message"
                ;;
        esac
    fi
}

error_exit() {
    log ERROR "$1"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            INSTALL_PROFILE="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --silent)
            SILENT=true
            shift
            ;;
        --no-cli-tools)
            NO_CLI_TOOLS=true
            shift
            ;;
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            cat <<EOF
Zero-Touch Installer for ipfs_accelerate_py

Usage: $0 [OPTIONS]

Options:
  --profile PROFILE       Installation profile (minimal|standard|full|cli)
  --cache-dir DIR        Cache directory (default: ~/.cache/ipfs_accelerate_py)
  --silent               Silent installation (no interactive prompts)
  --no-cli-tools         Skip CLI tool installation
  --skip-venv            Skip virtual environment creation
  --verify               Verify existing installation
  --verbose              Verbose output
  --help                 Show this help message

Examples:
  $0 --profile minimal
  $0 --profile standard --cache-dir /opt/cache
  $0 --silent --no-cli-tools

EOF
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# Banner
if [ "$SILENT" = false ]; then
    cat <<'EOF'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ipfs_accelerate_py Cache Infrastructure Installer          ║
║   Zero-Touch Multi-Platform Installation                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EOF
fi

log INFO "Starting installation with profile: $INSTALL_PROFILE"

# Detect platform and architecture
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case "$os" in
        linux*)
            PLATFORM="linux"
            ;;
        darwin*)
            PLATFORM="darwin"
            ;;
        freebsd*)
            PLATFORM="freebsd"
            ;;
        *)
            error_exit "Unsupported operating system: $os"
            ;;
    esac
    
    case "$arch" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        aarch64|arm64)
            ARCH="arm64"
            ;;
        armv7l|armhf)
            ARCH="armv7"
            ;;
        *)
            error_exit "Unsupported architecture: $arch"
            ;;
    esac
    
    log INFO "Detected platform: $PLATFORM ($ARCH)"
}

# Find Python
find_python() {
    for py in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$py" >/dev/null 2>&1; then
            PYTHON_CMD="$py"
            PYTHON_VERSION=$($py --version 2>&1 | awk '{print $2}')
            log INFO "Found Python: $PYTHON_CMD ($PYTHON_VERSION)"
            return 0
        fi
    done
    error_exit "Python 3.8+ not found. Please install Python first."
}

# Check if running in virtual environment
check_venv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        log INFO "Virtual environment detected: $VIRTUAL_ENV"
        PYTHON_CMD="python"
        return 0
    fi
    return 1
}

# Create virtual environment
create_venv() {
    if check_venv || [ "$SKIP_VENV" = true ]; then
        return 0
    fi
    
    log INFO "Creating virtual environment..."
    local venv_dir=".venv"
    
    if [ -d "$venv_dir" ]; then
        log WARN "Virtual environment already exists at $venv_dir"
        if [ "$SILENT" = false ]; then
            read -p "Use existing virtual environment? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$venv_dir"
                $PYTHON_CMD -m venv "$venv_dir" || error_exit "Failed to create virtual environment"
            fi
        fi
    else
        $PYTHON_CMD -m venv "$venv_dir" || error_exit "Failed to create virtual environment"
    fi
    
    # Activate venv
    source "$venv_dir/bin/activate" || error_exit "Failed to activate virtual environment"
    PYTHON_CMD="python"
    log SUCCESS "Virtual environment created and activated"
}

# Install Python dependencies
install_python_deps() {
    log INFO "Installing Python dependencies for profile: $INSTALL_PROFILE"
    
    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel || error_exit "Failed to upgrade pip"
    
    # Install based on profile
    case "$INSTALL_PROFILE" in
        minimal)
            $PYTHON_CMD -m pip install ipfs_accelerate_py[minimal] || error_exit "Failed to install minimal profile"
            ;;
        standard)
            $PYTHON_CMD -m pip install ipfs_accelerate_py[cache] || error_exit "Failed to install standard profile"
            ;;
        full)
            $PYTHON_CMD -m pip install ipfs_accelerate_py[all] || error_exit "Failed to install full profile"
            ;;
        cli)
            $PYTHON_CMD -m pip install ipfs_accelerate_py[cli] || error_exit "Failed to install CLI profile"
            ;;
        *)
            error_exit "Unknown profile: $INSTALL_PROFILE"
            ;;
    esac
    
    log SUCCESS "Python dependencies installed"
}

# Install CLI tools
install_cli_tools() {
    if [ "$NO_CLI_TOOLS" = true ]; then
        log INFO "Skipping CLI tools installation"
        return 0
    fi
    
    log INFO "Installing CLI tools..."
    
    # GitHub CLI
    if ! command -v gh >/dev/null 2>&1; then
        log INFO "Installing GitHub CLI..."
        case "$PLATFORM" in
            linux)
                if command -v apt-get >/dev/null 2>&1; then
                    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
                    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
                    sudo apt-get update && sudo apt-get install -y gh
                elif command -v yum >/dev/null 2>&1; then
                    sudo yum-config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
                    sudo yum install -y gh
                fi
                ;;
            darwin)
                if command -v brew >/dev/null 2>&1; then
                    brew install gh
                fi
                ;;
        esac
        log SUCCESS "GitHub CLI installed"
    else
        log INFO "GitHub CLI already installed"
    fi
    
    # HuggingFace CLI
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        log INFO "Installing HuggingFace CLI..."
        $PYTHON_CMD -m pip install huggingface-hub[cli] || log WARN "Failed to install HuggingFace CLI"
    else
        log INFO "HuggingFace CLI already installed"
    fi
    
    # Vast AI CLI
    if ! command -v vastai >/dev/null 2>&1; then
        log INFO "Installing Vast AI CLI..."
        $PYTHON_CMD -m pip install vastai || log WARN "Failed to install Vast AI CLI"
    else
        log INFO "Vast AI CLI already installed"
    fi
    
    # GitHub Copilot CLI (requires Node.js/npm)
    if command -v npm >/dev/null 2>&1; then
        if ! command -v github-copilot-cli >/dev/null 2>&1; then
            log INFO "Installing GitHub Copilot CLI..."
            npm install -g @githubnext/github-copilot-cli || log WARN "Failed to install GitHub Copilot CLI"
        else
            log INFO "GitHub Copilot CLI already installed"
        fi
    else
        log WARN "npm not found - skipping GitHub Copilot CLI installation"
    fi
    
    # OpenAI Codex CLI (requires Node.js/npm)
    if command -v npm >/dev/null 2>&1; then
        if ! npm list -g 2>/dev/null | grep -q "@openai/codex"; then
            log INFO "Installing OpenAI Codex CLI..."
            npm install -g @openai/codex 2>/dev/null || log WARN "Failed to install OpenAI Codex CLI (may not be publicly available)"
        else
            log INFO "OpenAI Codex CLI already installed"
        fi
    fi
    
    # Claude Code CLI (Anthropic)
    if ! command -v claude >/dev/null 2>&1; then
        log INFO "Installing Claude CLI..."
        $PYTHON_CMD -m pip install anthropic 2>/dev/null || log WARN "Failed to install Anthropic SDK (CLI wrapper available via our integration)"
    else
        log INFO "Claude CLI already installed"
    fi
    
    # Gemini CLI (Google)
    log INFO "Installing Google Generative AI SDK..."
    $PYTHON_CMD -m pip install google-generativeai 2>/dev/null || log WARN "Failed to install Google Generative AI SDK"
    
    # Groq CLI
    if ! command -v groq >/dev/null 2>&1; then
        log INFO "Installing Groq SDK..."
        $PYTHON_CMD -m pip install groq || log WARN "Failed to install Groq SDK"
    else
        log INFO "Groq SDK already installed"
    fi
    
    # VSCode CLI
    if ! command -v code >/dev/null 2>&1; then
        log WARN "VSCode CLI not found - install VSCode to get the 'code' command"
        log INFO "Visit: https://code.visualstudio.com/"
    else
        log SUCCESS "VSCode CLI already installed"
    fi
    
    log SUCCESS "CLI tools installation complete"
}

# Setup cache directory
setup_cache_dir() {
    log INFO "Setting up cache directory: $CACHE_DIR"
    mkdir -p "$CACHE_DIR" || error_exit "Failed to create cache directory"
    chmod 755 "$CACHE_DIR"
    log SUCCESS "Cache directory created"
}

# Configure environment
configure_env() {
    log INFO "Configuring environment variables..."
    
    local shell_rc=""
    if [ -n "$BASH_VERSION" ]; then
        shell_rc="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_rc="$HOME/.zshrc"
    fi
    
    if [ -n "$shell_rc" ] && [ -f "$shell_rc" ]; then
        # Check if already configured
        if ! grep -q "IPFS_ACCELERATE_CACHE_DIR" "$shell_rc"; then
            cat >> "$shell_rc" <<EOF

# ipfs_accelerate_py cache configuration
export IPFS_ACCELERATE_CACHE_DIR="$CACHE_DIR"
export IPFS_ACCELERATE_CACHE_TTL=3600
EOF
            log SUCCESS "Environment variables added to $shell_rc"
        else
            log INFO "Environment variables already configured"
        fi
    fi
}

# Verify installation
verify_installation() {
    log INFO "Verifying installation..."
    
    local errors=0
    
    # Test Python import
    if ! $PYTHON_CMD -c "from ipfs_accelerate_py.common.base_cache import BaseCache" 2>/dev/null; then
        log ERROR "Failed to import BaseCache"
        ((errors++))
    else
        log SUCCESS "BaseCache import successful"
    fi
    
    if ! $PYTHON_CMD -c "from ipfs_accelerate_py.common.cid_index import CIDIndex" 2>/dev/null; then
        log ERROR "Failed to import CIDIndex"
        ((errors++))
    else
        log SUCCESS "CIDIndex import successful"
    fi
    
    if ! $PYTHON_CMD -c "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache" 2>/dev/null; then
        log ERROR "Failed to import LLM cache"
        ((errors++))
    else
        log SUCCESS "LLM cache import successful"
    fi
    
    # Test CLI integrations
    if [ "$INSTALL_PROFILE" != "minimal" ]; then
        if ! $PYTHON_CMD -c "from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations" 2>/dev/null; then
            log ERROR "Failed to import CLI integrations"
            ((errors++))
        else
            log SUCCESS "CLI integrations import successful"
        fi
    fi
    
    # Test API integrations
    if [ "$INSTALL_PROFILE" = "standard" ] || [ "$INSTALL_PROFILE" = "full" ]; then
        if ! $PYTHON_CMD -c "from ipfs_accelerate_py.api_integrations import get_cached_openai_api" 2>/dev/null; then
            log ERROR "Failed to import API integrations"
            ((errors++))
        else
            log SUCCESS "API integrations import successful"
        fi
    fi
    
    # Check cache directory
    if [ ! -d "$CACHE_DIR" ]; then
        log ERROR "Cache directory not found: $CACHE_DIR"
        ((errors++))
    else
        log SUCCESS "Cache directory exists"
    fi
    
    if [ $errors -eq 0 ]; then
        log SUCCESS "All verification checks passed"
        return 0
    else
        log ERROR "$errors verification check(s) failed"
        return 1
    fi
}

# Main installation flow
main() {
    if [ "$VERIFY_ONLY" = true ]; then
        find_python
        verify_installation
        exit $?
    fi
    
    detect_platform
    find_python
    create_venv
    install_python_deps
    install_cli_tools
    setup_cache_dir
    configure_env
    verify_installation
    
    if [ "$SILENT" = false ]; then
        cat <<EOF

╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ✓ Installation Complete!                                   ║
║                                                              ║
║   Cache directory: $CACHE_DIR
║   Profile: $INSTALL_PROFILE
║   Platform: $PLATFORM ($ARCH)
║                                                              ║
║   Quick Start:                                               ║
║   $ python -c "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache; cache = get_global_llm_cache(); print('Cache ready!')"
║                                                              ║
║   Documentation: CLI_INTEGRATIONS.md                         ║
║   Logs: $LOG_FILE
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EOF
    fi
    
    log SUCCESS "Installation completed successfully"
}

# Run main installation
main
