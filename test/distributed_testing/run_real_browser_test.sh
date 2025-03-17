#!/bin/bash
# Run Real Browser Test for Selenium Browser Bridge
#
# This script sets up the environment and runs a real browser test with
# the Selenium Browser Bridge. It provides options for choosing browsers,
# models, platforms, and other settings.
#
# Usage:
#   ./run_real_browser_test.sh [options]

# Default options
BROWSER="chrome"
MODEL="bert-base-uncased"
PLATFORM="webgpu"
HEADLESS=true
ALLOW_SIMULATION=true
SETUP_WEBDRIVERS=true
VERBOSE=false
RESULTS_DIR="./reports"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --chrome)
      BROWSER="chrome"
      shift
      ;;
    --firefox)
      BROWSER="firefox"
      shift
      ;;
    --edge)
      BROWSER="edge"
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --bert)
      MODEL="bert-base-uncased"
      shift
      ;;
    --vit)
      MODEL="vit-base-patch16-224"
      shift
      ;;
    --whisper)
      MODEL="whisper-tiny"
      shift
      ;;
    --clip)
      MODEL="clip-vit-base-patch32"
      shift
      ;;
    --webgpu)
      PLATFORM="webgpu"
      shift
      ;;
    --webnn)
      PLATFORM="webnn"
      shift
      ;;
    --no-headless)
      HEADLESS=false
      shift
      ;;
    --no-simulation)
      ALLOW_SIMULATION=false
      shift
      ;;
    --no-setup)
      SETUP_WEBDRIVERS=false
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift
      shift
      ;;
    --detect-browsers)
      echo "Detecting available browsers..."
      if command -v google-chrome >/dev/null 2>&1 || command -v google-chrome-stable >/dev/null 2>&1; then
        echo "✅ Chrome detected"
      else
        echo "❌ Chrome not detected"
      fi
      if command -v firefox >/dev/null 2>&1; then
        echo "✅ Firefox detected"
      else
        echo "❌ Firefox not detected"
      fi
      if command -v microsoft-edge >/dev/null 2>&1 || command -v microsoft-edge-stable >/dev/null 2>&1; then
        echo "✅ Edge detected"
      else
        echo "❌ Edge not detected"
      fi
      exit 0
      ;;
    --help)
      echo "Run Real Browser Test for Selenium Browser Bridge"
      echo ""
      echo "Usage:"
      echo "  ./run_real_browser_test.sh [options]"
      echo ""
      echo "Options:"
      echo "  --chrome           Use Chrome browser"
      echo "  --firefox          Use Firefox browser"
      echo "  --edge             Use Edge browser"
      echo "  --model <model>    Set model name to test"
      echo "  --bert             Use BERT model"
      echo "  --vit              Use ViT model"
      echo "  --whisper          Use Whisper model"
      echo "  --clip             Use CLIP model"
      echo "  --webgpu           Use WebGPU platform"
      echo "  --webnn            Use WebNN platform"
      echo "  --no-headless      Show browser window (not headless)"
      echo "  --no-simulation    Disable simulation fallback"
      echo "  --no-setup         Skip WebDriver setup"
      echo "  --verbose          Enable verbose logging"
      echo "  --results-dir <dir> Set directory for result files"
      echo "  --detect-browsers  Detect available browsers"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Set up timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/real_browser_test_${BROWSER}_${PLATFORM}_${TIMESTAMP}.json"

# Set verbose logging if requested
if [ "$VERBOSE" = true ]; then
  export SELENIUM_BRIDGE_LOG_LEVEL=DEBUG
fi

# Setup WebDrivers if requested
if [ "$SETUP_WEBDRIVERS" = true ]; then
  echo "Setting up WebDrivers..."
  
  # Check for pip
  if ! command -v pip >/dev/null 2>&1; then
    echo "❌ pip not found. Please install pip first."
    exit 1
  fi
  
  # Install Selenium if not already installed
  if ! python -c "import selenium" >/dev/null 2>&1; then
    echo "Installing Selenium..."
    pip install selenium
  else
    echo "✅ Selenium already installed."
  fi
  
  # Install webdriver-manager for automatic WebDriver installation
  if ! python -c "import webdriver_manager" >/dev/null 2>&1; then
    echo "Installing webdriver-manager..."
    pip install webdriver-manager
  else
    echo "✅ webdriver-manager already installed."
  fi
  
  # Create a simple Python script to install WebDrivers
  cat > webdriver_setup.py << 'EOF'
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import os
import sys

# Function to check if a command exists
def command_exists(cmd):
    return os.system(f"command -v {cmd} > /dev/null 2>&1") == 0

# Install WebDrivers for detected browsers
if command_exists("google-chrome") or command_exists("google-chrome-stable"):
    try:
        print("Installing ChromeDriver...")
        ChromeDriverManager().install()
        print("✅ ChromeDriver installed successfully.")
    except Exception as e:
        print(f"❌ Error installing ChromeDriver: {str(e)}")

if command_exists("firefox"):
    try:
        print("Installing GeckoDriver...")
        GeckoDriverManager().install()
        print("✅ GeckoDriver installed successfully.")
    except Exception as e:
        print(f"❌ Error installing GeckoDriver: {str(e)}")

if command_exists("microsoft-edge") or command_exists("microsoft-edge-stable"):
    try:
        print("Installing EdgeDriver...")
        EdgeChromiumDriverManager().install()
        print("✅ EdgeDriver installed successfully.")
    except Exception as e:
        print(f"❌ Error installing EdgeDriver: {str(e)}")

print("WebDriver setup complete.")
EOF
  
  # Run the WebDriver setup script
  python webdriver_setup.py
  
  # Clean up the temporary script
  rm webdriver_setup.py
  
  echo "Setup complete."
fi

# Verify the browser exists
browser_exists=true
case $BROWSER in
  chrome)
    if ! command -v google-chrome >/dev/null 2>&1 && ! command -v google-chrome-stable >/dev/null 2>&1; then
      echo "❌ Chrome not detected."
      browser_exists=false
    fi
    ;;
  firefox)
    if ! command -v firefox >/dev/null 2>&1; then
      echo "❌ Firefox not detected."
      browser_exists=false
    fi
    ;;
  edge)
    if ! command -v microsoft-edge >/dev/null 2>&1 && ! command -v microsoft-edge-stable >/dev/null 2>&1; then
      echo "❌ Edge not detected."
      browser_exists=false
    fi
    ;;
esac

# If browser doesn't exist, use simulation mode
if [ "$browser_exists" = false ] && [ "$ALLOW_SIMULATION" = true ]; then
  echo "Browser not available. Using simulation mode."
  SIMULATION_FLAG=""
elif [ "$browser_exists" = false ] && [ "$ALLOW_SIMULATION" = false ]; then
  echo "❌ Browser not available and simulation mode is disabled. Exiting."
  exit 1
elif [ "$ALLOW_SIMULATION" = false ]; then
  SIMULATION_FLAG="--no-simulation"
else
  SIMULATION_FLAG=""
fi

# Set headless flag
if [ "$HEADLESS" = true ]; then
  HEADLESS_FLAG=""
else
  HEADLESS_FLAG="--no-headless"
fi

# Run the test
echo "Running real browser test with:"
echo "  Browser: $BROWSER"
echo "  Model: $MODEL"
echo "  Platform: $PLATFORM"
echo "  Headless: $HEADLESS"
echo "  Allow Simulation: $ALLOW_SIMULATION"
echo "  Results File: $RESULTS_FILE"
echo ""

# Execute the Python test script
python run_real_browser_test.py \
  --browser "$BROWSER" \
  --model "$MODEL" \
  --platform "$PLATFORM" \
  $HEADLESS_FLAG \
  $SIMULATION_FLAG \
  --save-results "$RESULTS_FILE"

# Check exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✅ Test completed successfully."
else
  echo "❌ Test failed with exit code $exit_code."
fi

exit $exit_code