#!/bin/bash
# iOS CI Runner Setup Script
#
# This script automates the setup of an iOS CI runner for IPFS Accelerate
# It performs environment checks, installs dependencies, configures the environment,
# and verifies device connectivity
#
# Usage:
#   ./setup_ios_ci_runner.sh [--device-id DEVICE_ID] [--simulator] [--register] [--token TOKEN] [--labels LABELS]
#
# Options:
#   --device-id DEVICE_ID  Specific iOS device ID (UDID) to use
#   --simulator            Use iOS simulator instead of physical device
#   --register             Register as GitHub Actions runner after setup
#   --token TOKEN          GitHub Actions runner registration token
#   --labels LABELS        Custom labels for the runner (comma-separated)
#   --help                 Display this help message
#
# Example:
#   ./setup_ios_ci_runner.sh --device-id 00008101-001D38810168001E --register --token ABC123
#
# Date: May 2025

set -e

# Default values
DEVICE_ID=""
USE_SIMULATOR=false
REGISTER=false
TOKEN=""
LABELS="ios,mobile,physical"
VERBOSE=true
RUNNER_VERSION="2.310.2"

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --simulator)
      USE_SIMULATOR=true
      LABELS="ios,mobile,simulator"
      shift
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
      echo "iOS CI Runner Setup Script"
      echo ""
      echo "Usage:"
      echo "  ./setup_ios_ci_runner.sh [--device-id DEVICE_ID] [--simulator] [--register] [--token TOKEN] [--labels LABELS]"
      echo ""
      echo "Options:"
      echo "  --device-id DEVICE_ID  Specific iOS device ID (UDID) to use"
      echo "  --simulator            Use iOS simulator instead of physical device"
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

# Ensure we're running on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
  error "This script must be run on macOS for iOS development"
fi

# Get macOS version
OS_VERSION="$(sw_vers -productVersion)"
log "macOS Version: $OS_VERSION"

# Check for Python
if ! check_command python3; then
  log "Installing Python..."
  
  # Install Homebrew if not present
  if ! check_command brew; then
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  
  brew install python@3.9
fi

# Check for Xcode command line tools
if ! check_command xcode-select; then
  log "Installing Xcode command line tools..."
  xcode-select --install
  
  # Wait for installation to complete
  log "Please complete the Xcode command line tools installation then press Enter to continue..."
  read
fi

# Check Xcode path
XCODE_PATH=$(xcode-select --print-path)
if [[ ! -d "$XCODE_PATH" ]]; then
  error "Xcode path not found: $XCODE_PATH. Please install Xcode from the App Store."
fi
log "Xcode path: $XCODE_PATH"

# Check for Git
if ! check_command git; then
  log "Installing Git..."
  brew install git
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
python3 -m pip install -r test/ios_test_harness/requirements.txt

# Install CoreML tools
log "Installing CoreML tools..."
python3 -m pip install coremltools>=6.0

# Configure environment
log "Configuring iOS environment..."
python3 test/setup_mobile_ci_runners.py --action configure --platform ios --verbose

# List connected devices
log "Checking for connected iOS devices..."
xcrun xctrace list devices

# If using simulator, list available simulators
if [ "$USE_SIMULATOR" = true ]; then
  log "Available iOS simulators:"
  xcrun simctl list
  
  # If no device ID specified, use the first available simulator
  if [ -z "$DEVICE_ID" ]; then
    # Find a booted simulator or boot one
    SIMULATOR_INFO=$(xcrun simctl list | grep "iPhone" | grep -v "unavailable" | head -1)
    if [[ $SIMULATOR_INFO =~ \(([A-Za-z0-9-]+)\) ]]; then
      DEVICE_ID="${BASH_REMATCH[1]}"
      log "Selected simulator: $DEVICE_ID"
    else
      error "No iOS simulators found. Please create one in Xcode."
    fi
    
    # Check if simulator is booted
    if ! xcrun simctl list | grep "$DEVICE_ID" | grep -q "Booted"; then
      log "Booting simulator $DEVICE_ID..."
      xcrun simctl boot "$DEVICE_ID"
    fi
  else
    # Boot the specified simulator if not already booted
    if ! xcrun simctl list | grep "$DEVICE_ID" | grep -q "Booted"; then
      log "Booting simulator $DEVICE_ID..."
      xcrun simctl boot "$DEVICE_ID" || error "Failed to boot simulator $DEVICE_ID"
    fi
  fi
fi

# Verify device connectivity
log "Verifying iOS device connectivity..."
if [ -n "$DEVICE_ID" ]; then
  if [ "$USE_SIMULATOR" = true ]; then
    python3 test/setup_mobile_ci_runners.py --action verify --platform ios --device-id "$DEVICE_ID" --verbose
  else
    # For physical devices
    python3 test/setup_mobile_ci_runners.py --action verify --platform ios --device-id "$DEVICE_ID" --verbose
  fi
else
  # Verify with first available device
  python3 test/setup_mobile_ci_runners.py --action verify --platform ios --verbose
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
    arm64)
      ARCH="arm64"
      ;;
    *)
      error "Unsupported architecture: $(uname -m)"
      ;;
  esac
  
  # Download runner package
  RUNNER_FILE="actions-runner-osx-${ARCH}-${RUNNER_VERSION}.tar.gz"
  
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
  
  # Add device specifics to labels
  if [ -n "$DEVICE_ID" ]; then
    # Try to get device name
    if [ "$USE_SIMULATOR" = true ]; then
      DEVICE_INFO=$(xcrun simctl list | grep "$DEVICE_ID" | head -1)
      if [[ $DEVICE_INFO =~ ([a-zA-Z0-9\ \(\)]+) ]]; then
        DEVICE_NAME="${BASH_REMATCH[1]}"
        DEVICE_NAME=$(echo "$DEVICE_NAME" | tr -d '()' | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
        LABELS="$LABELS,$DEVICE_NAME"
        log "Added device name to labels: $DEVICE_NAME"
      fi
    else
      # For physical devices
      DEVICE_INFO=$(xcrun xctrace list devices | grep "$DEVICE_ID")
      if [[ $DEVICE_INFO =~ ([a-zA-Z0-9\ ]+) ]]; then
        DEVICE_NAME="${BASH_REMATCH[1]}"
        DEVICE_NAME=$(echo "$DEVICE_NAME" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
        LABELS="$LABELS,$DEVICE_NAME"
        log "Added device name to labels: $DEVICE_NAME"
      fi
      
      # Try to get iOS version
      if [[ $DEVICE_INFO =~ iOS\ ([0-9\.]+) ]]; then
        IOS_VERSION="${BASH_REMATCH[1]}"
        IOS_VERSION=$(echo "$IOS_VERSION" | tr -d '.')
        LABELS="$LABELS,ios$IOS_VERSION"
        log "Added iOS version to labels: ios$IOS_VERSION"
      fi
    fi
  fi
  
  # Configure the runner
  ./config.sh --url https://github.com/yourusername/ipfs_accelerate_py --token "$TOKEN" --name "ios-$HOSTNAME" --labels "$LABELS" --unattended
  
  # Install as a service
  log "Installing runner as a service..."
  ./svc.sh install
  ./svc.sh start
  
  log "Runner successfully registered and started!"
  
  # Return to repo directory
  cd "$REPO_ROOT"
fi

# Run a test benchmark to confirm everything is working
log "Running a test benchmark..."
if [ -n "$DEVICE_ID" ]; then
  if [ "$USE_SIMULATOR" = true ]; then
    python3 test/ios_test_harness/run_ci_benchmarks.py --device-id "$DEVICE_ID" --simulator --output-db benchmark_results.duckdb --verbose
  else
    python3 test/ios_test_harness/run_ci_benchmarks.py --device-id "$DEVICE_ID" --output-db benchmark_results.duckdb --verbose
  fi
else
  log "No device ID specified for test benchmark. Skipping test run."
fi

log "iOS CI runner setup complete!"
log "For detailed instructions, see: test/MOBILE_CI_RUNNER_SETUP_GUIDE.md"
if [ "$REGISTER" = false ]; then
  log "To register this machine as a GitHub Actions runner, run this script with --register --token YOUR_TOKEN"
fi