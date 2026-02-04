#!/bin/bash
# Setup script for Selenium and WebDrivers, and running the comprehensive test suite
#
# This script:
# 1. Checks for and installs Selenium if needed
# 2. Installs WebDrivers for Chrome, Firefox, and Edge if available
# 3. Detects available browsers on the system
# 4. Runs the test suite with available browsers
# 5. Generates a comprehensive report of test results
#
# Usage:
#   ./setup_and_run_selenium_tests.sh [--skip-setup] [--quick] [--full] [--report-only]

# Parse arguments
SKIP_SETUP=false
TEST_MODE="standard"  # standard, quick, full, or report-only
REPORT_DIR="./reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/selenium_test_report_$TIMESTAMP.json"
SIMULATION_FLAG=""  # Flag to indicate simulation mode

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    --quick)
      TEST_MODE="quick"
      shift
      ;;
    --full)
      TEST_MODE="full"
      shift
      ;;
    --report-only)
      TEST_MODE="report-only"
      shift
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift
      shift
      ;;
    --simulate)
      SIMULATION_FLAG="--simulate"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./setup_and_run_selenium_tests.sh [--skip-setup] [--quick] [--full] [--report-only] [--report-dir <dir>] [--simulate]"
      exit 1
      ;;
  esac
done

# Create report directory if it doesn't exist
mkdir -p "$REPORT_DIR"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
  python -c "import $1" >/dev/null 2>&1
}

# Function to detect browsers
detect_browsers() {
  DETECTED_BROWSERS=""
  
  # Check for Chrome
  if command_exists google-chrome || command_exists google-chrome-stable; then
    DETECTED_BROWSERS="$DETECTED_BROWSERS chrome"
    echo "✅ Chrome detected"
  else
    echo "❌ Chrome not detected"
  fi
  
  # Check for Firefox
  if command_exists firefox; then
    DETECTED_BROWSERS="$DETECTED_BROWSERS firefox"
    echo "✅ Firefox detected"
  else
    echo "❌ Firefox not detected"
  fi
  
  # Check for Edge
  if command_exists microsoft-edge || command_exists microsoft-edge-stable; then
    DETECTED_BROWSERS="$DETECTED_BROWSERS edge"
    echo "✅ Edge detected"
  else
    echo "❌ Edge not detected"
  fi
  
  # Return detected browsers (trimmed)
  echo "${DETECTED_BROWSERS## }"
}

# Setup Selenium and WebDrivers if not skipped
if [ "$SKIP_SETUP" = false ]; then
  echo "Setting up Selenium and WebDrivers..."
  
  # Check for pip
  if ! command_exists pip; then
    echo "❌ pip not found. Please install pip first."
    exit 1
  fi
  
  # Install Selenium if not already installed
  if ! package_installed selenium; then
    echo "Installing Selenium..."
    pip install selenium
  else
    echo "✅ Selenium already installed."
  fi
  
  # Install webdriver-manager for automatic WebDriver installation
  if ! package_installed webdriver_manager; then
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
  echo "Setting up WebDrivers for detected browsers..."
  python webdriver_setup.py
  
  # Clean up the temporary script
  rm webdriver_setup.py
  
  echo "Setup complete."
else
  echo "Skipping setup as requested."
fi

# Detect browsers for testing
echo "Detecting available browsers..."
BROWSERS=$(detect_browsers)

if [ -z "$BROWSERS" ]; then
  echo "❌ No supported browsers detected. Running in simulation mode."
  BROWSERS="chrome"  # Default to Chrome for simulation
  SIMULATION_FLAG="--simulate"
else
  echo "✅ Detected browsers: $BROWSERS"
  SIMULATION_FLAG=""
fi

# If report-only mode, skip running tests
if [ "$TEST_MODE" = "report-only" ]; then
  echo "Report-only mode selected. Skipping test execution."
else
  # Run the test suite with detected browsers
  echo "Running Selenium integration tests..."
  
  if [ "$TEST_MODE" = "quick" ]; then
    echo "Running quick tests..."
    ./distributed_testing/run_selenium_integration_tests.sh --quick $SIMULATION_FLAG --save-report
  elif [ "$TEST_MODE" = "full" ]; then
    echo "Running full test suite..."
    ./distributed_testing/run_selenium_integration_tests.sh --full $SIMULATION_FLAG --save-report
  else
    echo "Running standard tests with browsers: $BROWSERS"
    # Convert spaces to commas for the --browsers argument
    BROWSER_ARG=$(echo "$BROWSERS" | tr ' ' ',')
    python distributed_testing/test_selenium_browser_integration.py --browsers "$BROWSER_ARG" --report-path "$REPORT_FILE" $SIMULATION_FLAG
  fi
  
  # Check test exit code
  if [ $? -eq 0 ]; then
    echo "✅ Tests completed successfully."
  else
    echo "❌ Tests failed with errors."
  fi
fi

# Generate report from the latest test results if not in quick or full mode
# (those modes generate reports through the run_selenium_integration_tests.sh script)
if [ "$TEST_MODE" != "quick" ] && [ "$TEST_MODE" != "full" ]; then
  echo "Generating test report..."
  
  # Check if the report file exists
  if [ -f "$REPORT_FILE" ]; then
    # Create a summary report using Python
    cat > generate_summary.py << EOF
import json
import sys
from datetime import datetime

try:
    # Load the test results
    with open("$REPORT_FILE", "r") as f:
        results = json.load(f)
    
    # Generate a summary
    summary = results.get("summary", {})
    
    # Create a markdown report
    markdown = f"""# Selenium Integration Test Report
    
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests:** {summary.get('total_tests', 0)}
- **Successful Tests:** {summary.get('successful_tests', 0)}
- **Failed Tests:** {summary.get('failed_tests', 0)}
- **Success Rate:** {summary.get('success_rate', 0) * 100:.2f}%

## Browser Success Rates

| Browser | Success Rate |
|---------|-------------|
"""
    
    # Add browser success rates
    for browser, rate in summary.get('browser_success_rates', {}).items():
        markdown += f"| {browser} | {rate * 100:.2f}% |\n"
    
    markdown += """
## Model Type Success Rates

| Model Type | Success Rate |
|------------|-------------|
"""
    
    # Add model type success rates
    for model, rate in summary.get('model_success_rates', {}).items():
        markdown += f"| {model} | {rate * 100:.2f}% |\n"
    
    markdown += """
## Recovery Statistics

"""
    
    # Add recovery statistics
    recovery_attempts = summary.get('total_recovery_attempts', 0)
    recovery_successes = summary.get('total_recovery_successes', 0)
    recovery_rate = summary.get('recovery_success_rate', 0)
    
    markdown += f"""- **Recovery Attempts:** {recovery_attempts}
- **Recovery Successes:** {recovery_successes}
- **Recovery Success Rate:** {recovery_rate * 100:.2f}%
"""
    
    # Write the markdown report
    report_md = "$REPORT_DIR/selenium_test_report_$TIMESTAMP.md"
    with open(report_md, "w") as f:
        f.write(markdown)
    
    print(f"✅ Summary report generated: {report_md}")
    
    # Print a basic summary to the console
    print("\n=== Test Summary ===")
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Successful Tests: {summary.get('successful_tests', 0)}")
    print(f"Failed Tests: {summary.get('failed_tests', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0) * 100:.2f}%")
    print(f"Recovery Attempts: {recovery_attempts}")
    print(f"Recovery Success Rate: {recovery_rate * 100:.2f}%")
    print("===================")
    
except Exception as e:
    print(f"❌ Error generating report: {str(e)}")
    sys.exit(1)
EOF
    
    # Run the report generation script
    python generate_summary.py
    
    # Clean up the temporary script
    rm generate_summary.py
  else
    echo "❌ No test report file found at $REPORT_FILE"
  fi
fi

echo "All tests and reporting complete."