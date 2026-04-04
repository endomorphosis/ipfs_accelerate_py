#!/bin/bash
# End-to-End Selenium Browser Recovery Test Runner
#
# This script provides a simple interface for running the comprehensive end-to-end
# Selenium browser recovery tests with various configurations.
#
# Usage:
#   ./run_selenium_e2e_tests.sh [options]
#
# Options:
#   --setup-only      Only set up the Selenium environment, don't run tests
#   --basic           Run basic test with Chrome only and a single model
#   --comprehensive   Run comprehensive test with all browsers and models
#   --chrome-only     Test only Chrome browser
#   --firefox-only    Test only Firefox browser
#   --edge-only       Test only Edge browser
#   --text-only       Test only text models (BERT)
#   --vision-only     Test only vision models (ViT)
#   --audio-only      Test only audio models (Whisper)
#   --multimodal-only Test only multimodal models (CLIP)
#   --no-failures     Run without failure injection
#   --simulate        Run in simulation mode
#   --clean-reports   Clean up old reports before running
#   --help            Show this help message

# Set default options
BROWSER="chrome"
MODEL="bert-base-uncased"
TEST_COUNT=3
FAILURES=""
SIMULATE=""
REPORT_PATH=""
RUN_TESTS=true
CLEAN_REPORTS=false
SETUP_SELENIUM=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --setup-only)
      RUN_TESTS=false
      shift
      ;;
    --basic)
      BROWSER="chrome"
      MODEL="bert-base-uncased"
      TEST_COUNT=2
      shift
      ;;
    --comprehensive)
      BROWSER="chrome,firefox,edge"
      MODEL="bert-base-uncased,vit-base-patch16-224,whisper-tiny,clip-vit-base-patch32"
      TEST_COUNT=3
      shift
      ;;
    --chrome-only)
      BROWSER="chrome"
      shift
      ;;
    --firefox-only)
      BROWSER="firefox"
      shift
      ;;
    --edge-only)
      BROWSER="edge"
      shift
      ;;
    --text-only)
      MODEL="bert-base-uncased"
      shift
      ;;
    --vision-only)
      MODEL="vit-base-patch16-224"
      shift
      ;;
    --audio-only)
      MODEL="whisper-tiny"
      shift
      ;;
    --multimodal-only)
      MODEL="clip-vit-base-patch32"
      shift
      ;;
    --all)
      BROWSER="chrome,firefox,edge"
      MODEL="bert-base-uncased,vit-base-patch16-224,whisper-tiny,clip-vit-base-patch32"
      shift
      ;;
    --test-count)
      TEST_COUNT="$2"
      shift
      shift
      ;;
    --no-failures)
      FAILURES="--no-failures"
      shift
      ;;
    --simulate)
      SIMULATE="--simulate"
      shift
      ;;
    --clean-reports)
      CLEAN_REPORTS=true
      shift
      ;;
    --no-setup)
      SETUP_SELENIUM=false
      shift
      ;;
    --help)
      echo "End-to-End Selenium Browser Recovery Test Runner"
      echo ""
      echo "Usage:"
      echo "  ./run_selenium_e2e_tests.sh [options]"
      echo ""
      echo "Options:"
      echo "  --setup-only      Only set up the Selenium environment, don't run tests"
      echo "  --basic           Run basic test with Chrome only and a single model"
      echo "  --comprehensive   Run comprehensive test with all browsers and models"
      echo "  --chrome-only     Test only Chrome browser"
      echo "  --firefox-only    Test only Firefox browser"
      echo "  --edge-only       Test only Edge browser"
      echo "  --text-only       Test only text models (BERT)"
      echo "  --vision-only     Test only vision models (ViT)"
      echo "  --audio-only      Test only audio models (Whisper)"
      echo "  --multimodal-only Test only multimodal models (CLIP)"
      echo "  --all             Test all browser-model combinations"
      echo "  --test-count N    Number of test iterations per combination"
      echo "  --no-failures     Run without failure injection"
      echo "  --simulate        Run in simulation mode"
      echo "  --clean-reports   Clean up old reports before running"
      echo "  --no-setup        Skip Selenium setup"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set Python virtual environment
VENV_DIR="../../.venv"
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated Python virtual environment from $VENV_DIR"
else
  echo "Virtual environment not found. Continuing with system Python..."
fi

# Create reports directory
REPORTS_DIR="./reports"
mkdir -p "$REPORTS_DIR"

# Clean reports if requested
if [ "$CLEAN_REPORTS" = true ]; then
  echo "Cleaning up old reports..."
  rm -f "$REPORTS_DIR"/*.json
  rm -f "$REPORTS_DIR"/*.md
fi

# Generate timestamp for report
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_PATH="--report-path $REPORTS_DIR/selenium_e2e_test_$TIMESTAMP.json"

# Set up Selenium environment
if [ "$SETUP_SELENIUM" = true ]; then
  echo "Setting up Selenium environment..."
  
  # Check if selenium is already installed
  if python -c "import selenium" &>/dev/null; then
    echo "✅ Selenium is already installed"
  else
    echo "Installing Selenium..."
    pip install selenium
  fi
  
  # Check if webdriver_manager is already installed
  if python -c "import webdriver_manager" &>/dev/null; then
    echo "✅ Webdriver Manager is already installed"
  else
    echo "Installing Webdriver Manager..."
    pip install webdriver_manager
  fi
  
  # Create a temporary Python script to install WebDrivers
  cat > webdriver_setup.py << 'EOF'
try:
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
            print("✅ ChromeDriver installed successfully")
        except Exception as e:
            print(f"⚠️ Error installing ChromeDriver: {str(e)}")

    if command_exists("firefox"):
        try:
            print("Installing GeckoDriver...")
            GeckoDriverManager().install()
            print("✅ GeckoDriver installed successfully")
        except Exception as e:
            print(f"⚠️ Error installing GeckoDriver: {str(e)}")

    if command_exists("microsoft-edge") or command_exists("microsoft-edge-stable"):
        try:
            print("Installing EdgeDriver...")
            EdgeChromiumDriverManager().install()
            print("✅ EdgeDriver installed successfully")
        except Exception as e:
            print(f"⚠️ Error installing EdgeDriver: {str(e)}")

    print("WebDriver setup complete")
except ImportError:
    print("⚠️ Selenium or webdriver_manager not installed. Install first with pip.")
EOF

  # Run the WebDriver setup script
  python webdriver_setup.py

  # Clean up temporary script
  rm webdriver_setup.py
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
  echo "Running Selenium E2E Browser Recovery Tests..."
  echo "Browsers: $BROWSER"
  echo "Models: $MODEL"
  echo "Test Count: $TEST_COUNT"
  
  if [ -n "$FAILURES" ]; then
    echo "Failure Injection: Disabled"
  else
    echo "Failure Injection: Enabled"
  fi
  
  if [ -n "$SIMULATE" ]; then
    echo "Simulation Mode: Enabled"
  else
    echo "Simulation Mode: Disabled"
  fi
  
  echo "Report Path: $REPORTS_DIR/selenium_e2e_test_$TIMESTAMP.json"
  echo ""
  echo "Starting tests..."
  
  # Run the Python test script
  python selenium_e2e_browser_recovery_test.py \
    --browser "$BROWSER" \
    --model "$MODEL" \
    --test-count "$TEST_COUNT" \
    $FAILURES \
    $SIMULATE \
    $REPORT_PATH
  
  TEST_EXIT_CODE=$?
  
  if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Tests completed successfully!"
    
    # Open the report if available and we're not in a CI environment
    SUMMARY_PATH="$REPORTS_DIR/recovery_test_summary.md"
    if [ -f "$SUMMARY_PATH" ] && [ -z "$CI" ]; then
      if command -v xdg-open &>/dev/null; then
        echo "Opening test summary report..."
        xdg-open "$SUMMARY_PATH" &>/dev/null
      elif command -v open &>/dev/null; then
        echo "Opening test summary report..."
        open "$SUMMARY_PATH" &>/dev/null
      else
        echo "Test summary available at: $SUMMARY_PATH"
      fi
    fi
  else
    echo "❌ Tests failed with exit code $TEST_EXIT_CODE"
  fi
fi

echo "Done!"
exit $TEST_EXIT_CODE