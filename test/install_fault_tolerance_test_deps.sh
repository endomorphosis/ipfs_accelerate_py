#!/bin/bash
# Install dependencies for fault tolerance testing with real browsers

set -e  # Exit on error

echo "Installing dependencies for fault tolerance testing..."

# Ensure pip is up to date
pip install --upgrade pip

# Install Python dependencies
pip install selenium webdriver-manager duckdb asyncio pytest pytest-asyncio

# Check if Chrome is installed
if ! command -v google-chrome &> /dev/null; then
    echo "Chrome not detected, installing Chrome..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
        sudo apt-get update
        sudo apt-get install -y google-chrome-stable
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        echo "Please install Chrome manually on macOS"
    else
        echo "Unsupported OS for automatic Chrome installation. Please install Chrome manually."
    fi
fi

# Check if Firefox is installed
if ! command -v firefox &> /dev/null; then
    echo "Firefox not detected, installing Firefox..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        sudo apt-get update
        sudo apt-get install -y firefox
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        echo "Please install Firefox manually on macOS"
    else
        echo "Unsupported OS for automatic Firefox installation. Please install Firefox manually."
    fi
fi

# Check if Edge is installed (most systems will need manual installation)
if ! command -v msedge &> /dev/null; then
    echo "Microsoft Edge not detected. You may need to install it manually."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "For Linux, you can install Edge using:"
        echo "curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg"
        echo "sudo install -o root -g root -m 644 microsoft.gpg /usr/share/keyrings/"
        echo "sudo sh -c 'echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/edge stable main\" > /etc/apt/sources.list.d/microsoft-edge.list'"
        echo "sudo apt update"
        echo "sudo apt install microsoft-edge-stable"
    fi
fi

# Create a browser test page for WebGPU/WebNN
mkdir -p test_pages
cat > test_pages/webgpu_webnn_test.html << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <title>WebGPU/WebNN Test Page</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .supported { background-color: #d4edda; color: #155724; }
        .not-supported { background-color: #f8d7da; color: #721c24; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }
    </style>
</head>
<body>
    <h1>WebGPU/WebNN Browser Capability Test</h1>
    
    <h2>WebGPU Status</h2>
    <div id="webgpu-status" class="status"></div>
    <div id="webgpu-adapter-info"></div>
    
    <h2>WebNN Status</h2>
    <div id="webnn-status" class="status"></div>
    
    <h2>Browser Information</h2>
    <pre id="browser-info"></pre>
    
    <script>
        // Display browser information
        document.getElementById('browser-info').textContent = 
            `User Agent: ${navigator.userAgent}\n` +
            `Platform: ${navigator.platform}\n` +
            `Vendor: ${navigator.vendor}\n` +
            `Languages: ${navigator.languages.join(', ')}\n` +
            `Hardware Concurrency: ${navigator.hardwareConcurrency}\n` +
            `Device Memory: ${navigator.deviceMemory || 'Not available'}GB`;
        
        // Check WebGPU support
        if ('gpu' in navigator) {
            document.getElementById('webgpu-status').textContent = 'WebGPU is supported in this browser!';
            document.getElementById('webgpu-status').className = 'status supported';
            
            // Get adapter info
            async function getAdapterInfo() {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        const info = await adapter.requestAdapterInfo();
                        document.getElementById('webgpu-adapter-info').innerHTML = 
                            `<strong>Adapter Info:</strong><br>` +
                            `Vendor: ${info.vendor || 'Not available'}<br>` +
                            `Architecture: ${info.architecture || 'Not available'}<br>` +
                            `Device: ${info.device || 'Not available'}<br>` +
                            `Description: ${info.description || 'Not available'}`;
                    } else {
                        document.getElementById('webgpu-adapter-info').textContent = 'No WebGPU adapter available';
                    }
                } catch (error) {
                    document.getElementById('webgpu-adapter-info').textContent = `Error getting adapter info: ${error.message}`;
                }
            }
            getAdapterInfo();
        } else {
            document.getElementById('webgpu-status').textContent = 'WebGPU is not supported in this browser';
            document.getElementById('webgpu-status').className = 'status not-supported';
        }
        
        // Check WebNN support
        if ('ml' in navigator && 'getNeuralNetworkContext' in navigator.ml) {
            document.getElementById('webnn-status').textContent = 'WebNN is supported in this browser!';
            document.getElementById('webnn-status').className = 'status supported';
        } else {
            document.getElementById('webnn-status').textContent = 'WebNN is not supported in this browser';
            document.getElementById('webnn-status').className = 'status not-supported';
        }
    </script>
</body>
</html>
EOL

echo "Created test page at test_pages/webgpu_webnn_test.html"

# Create a README for the fault tolerance tests
cat > FAULT_TOLERANCE_TESTING_README.md << 'EOL'
# Fault Tolerance Testing Guide

This guide explains how to run fault tolerance tests with real browsers to validate the cross-browser model sharding recovery mechanisms.

## Prerequisites

1. Ensure you have the required dependencies installed by running:
   ```
   ./install_fault_tolerance_test_deps.sh
   ```

2. Make sure you have at least one of the following browsers installed:
   - Google Chrome
   - Mozilla Firefox
   - Microsoft Edge

3. For comprehensive testing, selenium and webdriver-manager are required:
   ```
   pip install selenium webdriver-manager
   ```

## Running Tests

### Basic Test (Single Browser Type)

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome
```

### Testing with Multiple Browser Types

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome,firefox,edge
```

### Testing Fault Tolerance with Forced Failures

```bash
# Force a random browser failure during testing
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure

# Specify a failure type
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --failure-type crash

# Available failure types:
# - crash: Force browser to crash
# - hang: Force browser to hang (infinite loop)
# - memory: Force memory pressure by allocating large arrays
# - disconnect: Simulate network disconnection
```

### Testing with Different Recovery Strategies

```bash
# Test progressive recovery (tries simple strategies first, then more complex ones)
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --recovery-strategy progressive

# Other recovery strategies:
# - restart: Restart the failed browser
# - reconnect: Attempt to reconnect to the browser
# - failover: Switch to another browser
# - parallel: Try multiple strategies in parallel
```

### Testing Different Model Types

```bash
# Test text models
python test_real_browser_fault_tolerance.py --model bert-base-uncased --model-type text

# Test audio models (Firefox preferred for compute shader support)
python test_real_browser_fault_tolerance.py --model whisper-tiny --model-type audio

# Test vision models
python test_real_browser_fault_tolerance.py --model vit-base-patch16-224 --model-type vision

# Test multimodal models
python test_real_browser_fault_tolerance.py --model clip-vit-base-patch32 --model-type multimodal
```

### Performance Benchmarking After Recovery

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --benchmark --iterations 20
```

### Debugging

To see the browser windows during testing (disable headless mode):

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --show-browsers
```

For more detailed logs:

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --verbose
```

### Comprehensive Tests

For complete testing including browser failure, recovery, and performance benchmarking:

```bash
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome,firefox,edge --force-failure --benchmark --iterations 10 --output test_results.json
```

## Test Results

If an output file is specified with `--output`, test results will be saved in JSON format. Results include:

- Basic test information (model, browsers, configuration)
- Phases (each phase with status, duration, and results)
- For failure tests, detailed recovery information
- For benchmark tests, performance statistics
- Detailed execution timing and browser capabilities

## Troubleshooting

- If browser drivers are not found, you may need to install them manually or ensure webdriver-manager is working correctly
- On Linux, you may need to install additional dependencies for headless browser testing
- If WebGPU/WebNN are not detected, ensure your browsers are up-to-date
- For Firefox compute shader support issues, set MOZ_WEBGPU_ADVANCED_COMPUTE=1 environment variable
EOL

echo "Created FAULT_TOLERANCE_TESTING_README.md"

# Create a comprehensive test runner script
cat > run_comprehensive_fault_tolerance_tests.sh << 'EOL'
#!/bin/bash
# Run comprehensive fault tolerance tests across different models and browsers

# Set up output directory
OUTPUT_DIR="test_results/fault_tolerance/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Function to run a test and append to summary
run_test() {
    local model="$1"
    local model_type="$2"
    local browsers="$3"
    local failure_type="$4"
    local recovery_strategy="$5"
    local test_name="${model}_${model_type}_${browsers// /_}_${failure_type}_${recovery_strategy}"
    local output_file="${OUTPUT_DIR}/${test_name}.json"
    
    echo "----------------------------------------"
    echo "Running test: $test_name"
    echo "Model: $model ($model_type)"
    echo "Browsers: $browsers"
    echo "Failure type: $failure_type"
    echo "Recovery strategy: $recovery_strategy"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    python test_real_browser_fault_tolerance.py \
        --model "$model" \
        --model-type "$model_type" \
        --browsers "$browsers" \
        --force-failure \
        --failure-type "$failure_type" \
        --recovery-strategy "$recovery_strategy" \
        --benchmark \
        --iterations 5 \
        --output "$output_file"
    
    local status=$?
    
    # Append to summary
    echo "| $model | $model_type | $browsers | $failure_type | $recovery_strategy | $([ $status -eq 0 ] && echo "✅ Pass" || echo "❌ Fail") |" >> "${OUTPUT_DIR}/summary.md"
    
    # Wait a bit to let resources clean up
    sleep 5
}

# Create summary file header
echo "# Fault Tolerance Test Results - $(date)" > "${OUTPUT_DIR}/summary.md"
echo "" >> "${OUTPUT_DIR}/summary.md"
echo "| Model | Type | Browsers | Failure | Recovery | Status |" >> "${OUTPUT_DIR}/summary.md"
echo "|-------|------|----------|---------|----------|--------|" >> "${OUTPUT_DIR}/summary.md"

# Define test cases
# Text models
run_test "bert-base-uncased" "text" "chrome,firefox" "crash" "progressive"
run_test "bert-base-uncased" "text" "chrome,edge" "memory" "failover"

# Audio models
if [ -n "$RUN_AUDIO_TESTS" ]; then
    run_test "whisper-tiny" "audio" "firefox,chrome" "disconnect" "progressive"
    run_test "whisper-tiny" "audio" "firefox,edge" "hang" "restart"
fi

# Vision models
run_test "vit-base-patch16-224" "vision" "chrome,firefox" "crash" "progressive" 
run_test "vit-base-patch16-224" "vision" "chrome,edge" "disconnect" "parallel"

# Multimodal models
if [ -n "$RUN_MULTIMODAL_TESTS" ]; then
    run_test "clip-vit-base-patch32" "multimodal" "chrome,firefox,edge" "memory" "progressive"
fi

echo "Tests completed. Results saved to: ${OUTPUT_DIR}"
echo "Summary available at: ${OUTPUT_DIR}/summary.md"
EOL

chmod +x run_comprehensive_fault_tolerance_tests.sh
echo "Created run_comprehensive_fault_tolerance_tests.sh"

echo "Installation complete!"
echo "To run fault tolerance tests, see FAULT_TOLERANCE_TESTING_README.md"