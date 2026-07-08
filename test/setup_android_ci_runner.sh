#!/bin/bash
# Android CI Runner Setup Script
#
# This script automates the setup of an Android CI runner for IPFS Accelerate
# It performs environment checks, installs dependencies, configures the environment,
# and verifies device connectivity
#
# Usage:
#   ./setup_android_ci_runner.sh [--device-id DEVICE_ID] [--register] [--token TOKEN] [--labels LABELS]
#
# Options:
#   --device-id DEVICE_ID  Specific Android device ID to use
#   --register             Register as GitHub Actions runner after setup
#   --token TOKEN          GitHub Actions runner registration token
#   --labels LABELS        Custom labels for the runner (comma-separated)
#   --help                 Display this help message
#
# Example:
#   ./setup_android_ci_runner.sh --device-id emulator-5554 --register --token ABC123
#
# Date: May 2025

set -e

# Default values
DEVICE_ID=""
REGISTER=false
TOKEN=""
LABELS="android,mobile,physical"
VERBOSE=true
RUNNER_VERSION="2.310.2"

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --register)
      REGISTER=true
      shift
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --labels)
      LABELS="$2"
      shift 2
      ;;
    --runner-version)
      RUNNER_VERSION="$2"
      shift 2
      ;;
    --help)
      echo "Android CI Runner Setup Script"
      echo ""
      echo "Usage:"
      echo "  ./setup_android_ci_runner.sh [--device-id DEVICE_ID] [--register] [--token TOKEN] [--labels LABELS]"
      echo ""
      echo "Options:"
      echo "  --device-id DEVICE_ID  Specific Android device ID to use"
      echo "  --register             Register as GitHub Actions runner after setup"
      echo "  --token TOKEN          GitHub Actions runner registration token"
      echo "  --labels LABELS        Custom labels for the runner (comma-separated)"
      echo "  --runner-version VERSION GitHub Actions runner version (default: 2.310.2)"
      echo "  --help                 Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Helper functions
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
  exit 1
}

check_command() {
  if ! command -v $1 &> /dev/null; then
    log "Command '$1' not found. Installing..."
    return 1
  else
    log "Command '$1' is installed: $(which $1)"
    return 0
  fi
}

# Check if running as root and get sudo if needed
if [[ $EUID -ne 0 ]]; then
  if ! command -v sudo &> /dev/null; then
    error "This script requires sudo but it's not installed. Please install sudo or run as root."
  fi
  SUDO="sudo"
else
  SUDO=""
fi

# Detect OS
OS="$(uname -s)"
case "${OS}" in
  Linux*)
    log "Detected Linux system"
    if [ -f /etc/os-release ]; then
      . /etc/os-release
      OS_NAME="$NAME"
      OS_VERSION="$VERSION_ID"
      log "Distribution: $OS_NAME $OS_VERSION"
    else
      log "Unknown Linux distribution"
      OS_NAME="Linux"
    fi
    ;;
  Darwin*)
    log "Detected macOS system"
    OS_NAME="macOS"
    OS_VERSION="$(sw_vers -productVersion)"
    log "Version: $OS_VERSION"
    ;;
  *)
    error "Unsupported operating system: $OS"
    ;;
esac

# Check for Python
if ! check_command python3; then
  log "Installing Python..."
  case "${OS_NAME}" in
    Ubuntu|Debian*)
      $SUDO apt-get update
      $SUDO apt-get install -y python3 python3-pip python3-venv
      ;;
    macOS)
      if ! check_command brew; then
        log "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      fi
      brew install python@3.9
      ;;
    *)
      error "Cannot install Python on this OS automatically. Please install Python 3.9+ manually."
      ;;
  esac
fi

# Check for Java
if ! check_command java; then
  log "Installing Java..."
  case "${OS_NAME}" in
    Ubuntu|Debian*)
      $SUDO apt-get update
      $SUDO apt-get install -y openjdk-11-jdk
      ;;
    macOS)
      brew install openjdk@11
      ;;
    *)
      error "Cannot install Java on this OS automatically. Please install Java 11+ manually."
      ;;
  esac
fi

# Check for ADB
if ! check_command adb; then
  log "Installing Android Debug Bridge..."
  case "${OS_NAME}" in
    Ubuntu|Debian*)
      $SUDO apt-get update
      $SUDO apt-get install -y android-tools-adb
      ;;
    macOS)
      brew install android-platform-tools
      ;;
    *)
      error "Cannot install ADB on this OS automatically. Please install Android SDK platform tools manually."
      ;;
  esac
fi

# Check for Git
if ! check_command git; then
  log "Installing Git..."
  case "${OS_NAME}" in
    Ubuntu|Debian*)
      $SUDO apt-get update
      $SUDO apt-get install -y git
      ;;
    macOS)
      brew install git
      ;;
    *)
      error "Cannot install Git on this OS automatically. Please install Git manually."
      ;;
  esac
fi

# Set up Python environment
log "Setting up Python environment..."
python3 -m pip install --upgrade pip

# Find the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -f "$SCRIPT_DIR/../ipfs_accelerate_py.py" ]]; then
  REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
else
  # Try to locate the repository
  if [[ -d "ipfs_accelerate_py" ]]; then
    REPO_ROOT="$( pwd )/ipfs_accelerate_py"
  else
    log "Repository not found. Cloning..."
    git clone https://github.com/yourusername/ipfs_accelerate_py.git
    REPO_ROOT="$( pwd )/ipfs_accelerate_py"
  fi
fi

log "Repository root: $REPO_ROOT"
cd "$REPO_ROOT"

# Install Python dependencies
log "Installing project dependencies..."
python3 -m pip install -r requirements.txt
python3 -m pip install -r test/android_test_harness/requirements.txt

# Configure environment
log "Configuring Android environment..."
python3 test/setup_mobile_ci_runners.py --action configure --platform android --verbose

# List connected devices
log "Checking for connected Android devices..."
adb devices -l

# Restart ADB server
log "Restarting ADB server..."
adb kill-server
adb start-server

sleep 2

# List devices again
log "Connected Android devices:"
adb devices -l

# Verify device connectivity
if [ -n "$DEVICE_ID" ]; then
  log "Verifying connectivity to device: $DEVICE_ID"
  python3 test/setup_mobile_ci_runners.py --action verify --platform android --device-id "$DEVICE_ID" --verbose
else
  log "Verifying connectivity to all connected devices..."
  python3 test/setup_mobile_ci_runners.py --action verify --platform android --verbose
fi

# Install workflows
log "Installing CI workflows..."
python3 test/setup_ci_workflows.py --install --verbose

# Register as GitHub Actions runner if requested
if [ "$REGISTER" = true ]; then
  if [ -z "$TOKEN" ]; then
    error "Registration token is required for runner registration. Use --token TOKEN."
  fi
  
  log "Registering as GitHub Actions runner..."
  
  # Create actions-runner directory
  mkdir -p ~/actions-runner && cd ~/actions-runner
  
  # Determine architecture
  case "$(uname -m)" in
    x86_64)
      ARCH="x64"
      ;;
    arm64|aarch64)
      ARCH="arm64"
      ;;
    *)
      error "Unsupported architecture: $(uname -m)"
      ;;
  esac
  
  # Download runner package
  case "${OS}" in
    Linux*)
      RUNNER_FILE="actions-runner-linux-${ARCH}-${RUNNER_VERSION}.tar.gz"
      ;;
    Darwin*)
      RUNNER_FILE="actions-runner-osx-${ARCH}-${RUNNER_VERSION}.tar.gz"
      ;;
  esac
  
  if [ ! -f "$RUNNER_FILE" ]; then
    log "Downloading runner package..."
    curl -O -L "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_FILE}"
  fi
  
  log "Extracting runner package..."
  tar xzf "./${RUNNER_FILE}"
  
  # Get hostname for runner name
  HOSTNAME=$(hostname)
  
  # Configure the runner
  log "Configuring runner with labels: $LABELS"
  
  # Add device ID to labels if provided
  if [ -n "$DEVICE_ID" ]; then
    DEVICE_MODEL=$(adb -s "$DEVICE_ID" shell getprop ro.product.model 2>/dev/null | tr -d '\r' | tr ' ' '-' | tr -d "'" | tr '[:upper:]' '[:lower:]')
    if [ -n "$DEVICE_MODEL" ]; then
      LABELS="$LABELS,$DEVICE_MODEL"
      log "Added device model to labels: $DEVICE_MODEL"
    fi
    
    ANDROID_VERSION=$(adb -s "$DEVICE_ID" shell getprop ro.build.version.release 2>/dev/null | tr -d '\r' | tr -d "'" | tr -d '.')
    if [ -n "$ANDROID_VERSION" ]; then
      LABELS="$LABELS,android$ANDROID_VERSION"
      log "Added Android version to labels: android$ANDROID_VERSION"
    fi
  fi
  
  # Configure the runner
  ./config.sh --url https://github.com/yourusername/ipfs_accelerate_py --token "$TOKEN" --name "android-$HOSTNAME" --labels "$LABELS" --unattended
  
  # Install as a service
  log "Installing runner as a service..."
  
  case "${OS}" in
    Linux*)
      $SUDO ./svc.sh install
      $SUDO ./svc.sh start
      ;;
    Darwin*)
      ./svc.sh install
      ./svc.sh start
      ;;
  esac
  
  log "Runner successfully registered and started!"
  
  # Return to repo directory
  cd "$REPO_ROOT"
fi

# Run a test benchmark to confirm everything is working
log "Running a test benchmark..."
if [ -n "$DEVICE_ID" ]; then
  python3 test/android_test_harness/run_ci_benchmarks.py --device-id "$DEVICE_ID" --output-db benchmark_results.duckdb --verbose
else
  # Use first available device
  FIRST_DEVICE=$(adb devices | grep -v "List" | grep "device$" | head -1 | cut -f1)
  if [ -n "$FIRST_DEVICE" ]; then
    log "Using device: $FIRST_DEVICE"
    python3 test/android_test_harness/run_ci_benchmarks.py --device-id "$FIRST_DEVICE" --output-db benchmark_results.duckdb --verbose
  else
    log "No devices found for testing. Please connect a device and try again."
  fi
fi

log "Android CI runner setup complete!"
log "For detailed instructions, see: test/MOBILE_CI_RUNNER_SETUP_GUIDE.md"
if [ "$REGISTER" = false ]; then
  log "To register this machine as a GitHub Actions runner, run this script with --register --token YOUR_TOKEN"
fi