#!/bin/bash
# Run WebNN and WebGPU quantization tests with various configurations

# Default parameters
MODEL="bert-base-uncased"
HEADLESS=false

print_help() {
    echo "WebNN and WebGPU Quantization Test Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model MODEL       Model to test (default: bert-base-uncased)"
    echo "  --headless          Run in headless mode (default: false)"
    echo "  --all               Run all tests (WebGPU and WebNN, all browsers)"
    echo "  --webgpu-only       Run only WebGPU tests"
    echo "  --webnn-only        Run only WebNN tests"
    echo "  --chrome            Run tests with Chrome"
    echo "  --firefox           Run tests with Firefox"
    echo "  --edge              Run tests with Edge"
    echo "  --safari            Run tests with Safari"
    echo "  --mixed-precision   Enable mixed precision testing"
    echo "  --ultra-low-prec    Enable ultra-low precision (2-bit) testing"
    echo "  --experimental      Try experimental precision with WebNN (may fail with errors)"
    echo "  --help              Display this help"
    echo ""
    echo "Examples:"
    echo "  $0 --webgpu-only --chrome --model bert-base-uncased"
    echo "  $0 --webnn-only --edge --mixed-precision"
    echo "  $0 --all --headless"
    echo "  $0 --webgpu-only --firefox --ultra-low-prec"
    echo ""
}

# Check if we have any arguments
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --webgpu-only)
            WEBGPU_ONLY=true
            shift
            ;;
        --webnn-only)
            WEBNN_ONLY=true
            shift
            ;;
        --chrome)
            RUN_CHROME=true
            shift
            ;;
        --firefox)
            RUN_FIREFOX=true
            shift
            ;;
        --edge)
            RUN_EDGE=true
            shift
            ;;
        --safari)
            RUN_SAFARI=true
            shift
            ;;
        --mixed-precision)
            MIXED_PRECISION=true
            shift
            ;;
        --ultra-low-prec)
            ULTRA_LOW_PRECISION=true
            shift
            ;;
        --experimental)
            EXPERIMENTAL_PRECISION=true
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Set default browsers if none specified
if [ "$RUN_ALL" = true ] || [ -z "$RUN_CHROME" -a -z "$RUN_FIREFOX" -a -z "$RUN_EDGE" -a -z "$RUN_SAFARI" ]; then
    RUN_CHROME=true
    RUN_FIREFOX=true
    
    # Only run Edge on Windows
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        RUN_EDGE=true
    fi
    
    # Only run Safari on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        RUN_SAFARI=true
    fi
fi

# Set default platforms if none specified
if [ "$RUN_ALL" = true ] || [ -z "$WEBGPU_ONLY" -a -z "$WEBNN_ONLY" ]; then
    WEBGPU_ONLY=true
    WEBNN_ONLY=true
fi

# Create output directory
mkdir -p quantization_results

# Define headless option
HEADLESS_OPT=""
if [ "$HEADLESS" = true ]; then
    HEADLESS_OPT="--headless"
fi

# Define mixed precision option
MIXED_PRECISION_OPT=""
if [ "$MIXED_PRECISION" = true ]; then
    MIXED_PRECISION_OPT="--mixed-precision"
fi

# Define experimental precision option
EXPERIMENTAL_OPT=""
if [ "$EXPERIMENTAL_PRECISION" = true ]; then
    EXPERIMENTAL_OPT="--experimental-precision"
fi

echo "==== WebNN and WebGPU Quantization Tests ===="
echo "Model: $MODEL"
echo "Headless: $HEADLESS"
echo "Mixed precision: $MIXED_PRECISION"
echo "Experimental precision: $EXPERIMENTAL_PRECISION"
echo "Starting tests..."
echo ""

# Run WebGPU tests
if [ "$WEBGPU_ONLY" = true ]; then
    # Test WebGPU with Chrome
    if [ "$RUN_CHROME" = true ]; then
        echo "=== Testing WebGPU with Chrome ==="
        echo "Testing 4-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT
        
        # Test ultra-low precision (2-bit) if ULTRA_LOW_PRECISION is set
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            echo "Testing 2-bit ultra-low precision..."
            python test_webnn_webgpu_simplified.py --platform webgpu --browser chrome --model $MODEL --bits 2 $HEADLESS_OPT $MIXED_PRECISION_OPT
        fi
        echo ""
    fi
    
    # Test WebGPU with Firefox
    if [ "$RUN_FIREFOX" = true ]; then
        echo "=== Testing WebGPU with Firefox ==="
        echo "Testing 4-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser firefox --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser firefox --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT
        
        # Test ultra-low precision (2-bit) if ULTRA_LOW_PRECISION is set
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            echo "Testing 2-bit ultra-low precision..."
            python test_webnn_webgpu_simplified.py --platform webgpu --browser firefox --model $MODEL --bits 2 $HEADLESS_OPT $MIXED_PRECISION_OPT
        fi
        echo ""
    fi
    
    # Test WebGPU with Edge (Windows only)
    if [ "$RUN_EDGE" = true ]; then
        echo "=== Testing WebGPU with Edge ==="
        echo "Testing 4-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser edge --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser edge --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT
        
        # Test ultra-low precision (2-bit) if ULTRA_LOW_PRECISION is set
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            echo "Testing 2-bit ultra-low precision..."
            python test_webnn_webgpu_simplified.py --platform webgpu --browser edge --model $MODEL --bits 2 $HEADLESS_OPT $MIXED_PRECISION_OPT
        fi
        echo ""
    fi
    
    # Test WebGPU with Safari (macOS only)
    if [ "$RUN_SAFARI" = true ]; then
        echo "=== Testing WebGPU with Safari ==="
        echo "Testing 4-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser safari --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webgpu --browser safari --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT
        
        # Note: Safari may not support ultra-low precision well, so we don't test 2-bit by default
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            echo "Testing 2-bit ultra-low precision (experimental for Safari)..."
            python test_webnn_webgpu_simplified.py --platform webgpu --browser safari --model $MODEL --bits 2 $HEADLESS_OPT $MIXED_PRECISION_OPT
        fi
        echo ""
    fi
fi

# Run WebNN tests
if [ "$WEBNN_ONLY" = true ]; then
    # Test WebNN with Chrome
    if [ "$RUN_CHROME" = true ]; then
        echo "=== Testing WebNN with Chrome ==="
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webnn --browser chrome --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        
        # Optionally test with lower precision to demonstrate behavior
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            if [ "$EXPERIMENTAL_PRECISION" = true ]; then
                echo "Testing with experimental 4-bit request (will attempt 4-bit with expected errors)..."
            else
                echo "Testing with 4-bit request (should use 8-bit fallback)..."
            fi
            python test_webnn_webgpu_simplified.py --platform webnn --browser chrome --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        fi
        echo ""
    fi
    
    # Test WebNN with Edge (Windows only)
    if [ "$RUN_EDGE" = true ]; then
        echo "=== Testing WebNN with Edge ==="
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webnn --browser edge --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        
        # Optionally test with lower precision to demonstrate behavior
        if [ "$ULTRA_LOW_PRECISION" = true ]; then
            if [ "$EXPERIMENTAL_PRECISION" = true ]; then
                echo "Testing with experimental 4-bit request (will attempt 4-bit with expected errors)..."
            else
                echo "Testing with 4-bit request (should use 8-bit fallback)..."
            fi
            python test_webnn_webgpu_simplified.py --platform webnn --browser edge --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        fi
        echo ""
    fi
    
    # Test WebNN with Safari (macOS only)
    if [ "$RUN_SAFARI" = true ]; then
        echo "=== Testing WebNN with Safari ==="
        echo "Testing 8-bit quantization..."
        python test_webnn_webgpu_simplified.py --platform webnn --browser safari --model $MODEL --bits 8 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        
        # Optionally test with lower precision to demonstrate behavior
        if [ "$ULTRA_LOW_PRECISION" = true ] && [ "$EXPERIMENTAL_PRECISION" = true ]; then
            echo "Testing with experimental 4-bit request (will attempt 4-bit with expected errors)..."
            python test_webnn_webgpu_simplified.py --platform webnn --browser safari --model $MODEL --bits 4 $HEADLESS_OPT $MIXED_PRECISION_OPT $EXPERIMENTAL_OPT
        fi
        echo ""
    fi
fi

echo "==== All tests completed ===="