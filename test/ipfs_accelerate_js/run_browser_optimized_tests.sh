#!/bin/bash

# Script to run browser-specific optimization tests and benchmarks
# This script helps demonstrate the browser-specific quantization and operation fusion optimizations

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}WebGPU Browser-Specific Optimization Tests${NC}"
echo "==============================================="

# Check if http-server is installed
if ! [ -x "$(command -v http-server)" ]; then
  echo -e "${YELLOW}http-server is not installed, installing...${NC}"
  npm install -g http-server
fi

# Function to build the TypeScript files
build_project() {
  echo -e "${GREEN}Building project...${NC}"
  npm run build
}

# Function to run tests
run_tests() {
  echo -e "${GREEN}Running browser-specific shader tests...${NC}"
  npm test -- -t "browser.*shader"
  
  echo -e "${GREEN}Running fusion quantization tests...${NC}"
  npm test -- -t "fusion.*quantization"

  echo -e "${GREEN}Running browser-specific fusion integration tests...${NC}"
  npm test -- -t "browser.*fusion"
  
  echo -e "${GREEN}Running browser-specific elementwise operation tests...${NC}"
  echo -e "${YELLOW}Testing ReLU and Add operations...${NC}"
  npx ts-node test/browser_specific_elementwise_test.ts
  
  echo -e "${YELLOW}Testing Tanh and Sigmoid operations...${NC}"
  npx ts-node test/browser_specific_elementwise_advanced_test.ts
  
  echo -e "${YELLOW}Testing GELU operations and fusion...${NC}"
  npx ts-node test/browser_specific_gelu_test.ts
  
  echo -e "${YELLOW}Testing SiLU operations and fusion...${NC}"
  npx ts-node test/browser_specific_silu_test.ts
  
  echo -e "${YELLOW}Testing Leaky ReLU operations and fusion...${NC}"
  npx ts-node test/browser_specific_leaky_relu_test.ts
  
  echo -e "${YELLOW}Testing Layer Normalization operations and fusion...${NC}"
  npx ts-node test/layer_normalization_test.ts
}

# Function to start http-server for benchmark visualization
start_benchmark_server() {
  echo -e "${GREEN}Starting HTTP server for benchmark visualization...${NC}"
  echo -e "Open your browser at ${YELLOW}http://localhost:8080/examples/browser_specific_quantization_benchmark.html${NC}"
  http-server -p 8080 -c-1 .
}

# Process command line arguments
case "$1" in
  build)
    build_project
    ;;
  test)
    run_tests
    ;;
  benchmark)
    start_benchmark_server
    ;;
  all)
    build_project
    run_tests
    start_benchmark_server
    ;;
  *)
    echo "Usage: $0 {build|test|benchmark|all}"
    echo "  build      - Build TypeScript project"
    echo "  test       - Run browser-specific optimization tests"
    echo "  benchmark  - Start HTTP server for benchmark visualization"
    echo "  all        - Run build, tests, and start benchmark server"
    exit 1
    ;;
esac

echo -e "${GREEN}Done!${NC}"