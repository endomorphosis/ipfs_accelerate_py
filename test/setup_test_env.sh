#!/bin/bash
# Setup script for the IPFS Accelerate test environment

# Exit on error
set -e

echo "Setting up test environment..."

# Create directory structure if it doesn't exist
mkdir -p test/models/text/bert
mkdir -p test/models/text/t5
mkdir -p test/models/text/gpt
mkdir -p test/models/vision/vit
mkdir -p test/models/audio/whisper
mkdir -p test/hardware/webgpu/compute_shaders
mkdir -p test/hardware/webnn
mkdir -p test/hardware/cuda
mkdir -p test/hardware/cpu
mkdir -p test/api/llm_providers
mkdir -p test/api/huggingface
mkdir -p test/api/local_servers
mkdir -p test/integration/browser
mkdir -p test/integration/database
mkdir -p test/integration/distributed
mkdir -p test/common
mkdir -p test/docs

# Create pytest.ini if it doesn't exist
if [ ! -f "pytest.ini" ]; then
    cat > pytest.ini << 'EOL'
[pytest]
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    webgpu: mark test as requiring WebGPU
    webnn: mark test as requiring WebNN
    cuda: mark test as requiring CUDA
    rocm: mark test as requiring ROCm
    slow: mark test as slow
    model: mark test as a model test
    hardware: mark test as a hardware test
    api: mark test as an API test
    integration: mark test as an integration test
    distributed: mark test as a distributed test
    browser: mark test as requiring a browser
EOL
    echo "Created pytest.ini"
fi

# Create common/__init__.py if it doesn't exist
if [ ! -f "test/common/__init__.py" ]; then
    cat > test/common/__init__.py << 'EOL'
"""
Common utilities for IPFS Accelerate tests.

This package contains shared code, fixtures, and utilities used across
the test framework.
"""
EOL
    echo "Created test/common/__init__.py"
fi

# Create __init__.py files in all directories to make them proper packages
find test -type d -exec touch {}/__init__.py \;
echo "Created __init__.py files in all test directories"

# Check if required Python packages are installed
echo -e "\nChecking if required Python packages are installed..."
required_packages=("pytest" "pytest-html" "pytest-cov" "numpy")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "\nThe following required packages are missing:"
    for package in "${missing_packages[@]}"; do
        echo "  - $package"
    done
    
    echo -e "\nPlease install them using:"
    echo "pip install ${missing_packages[*]}"
else
    echo "All required packages are installed."
fi

# Finish
echo -e "\nTest environment setup complete!"
echo "To verify the environment, run:"
echo "python3 verify_test_environment.py"