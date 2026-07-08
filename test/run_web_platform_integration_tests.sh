#!/bin/bash
# Web Platform Integration Test Runner (April 2025)
# This script runs comprehensive tests for web platform integration with all optimizations
# including April 2025 enhancements for 4-bit quantization and memory optimization

# Set options
set -e  # Exit on error

# Default values
MODELS=""
USE_BROWSER_AUTOMATION=false
BROWSER="chrome"
USE_FIREFOX=false

# March 2025 optimizations
ENABLE_COMPUTE_SHADERS=true
ENABLE_SHADER_PRECOMPILE=true
ENABLE_PARALLEL_LOADING=true
USE_FIREFOX_ADVANCED_COMPUTE=false  # Firefox's exceptional performance with --MOZ_WEBGPU_ADVANCED_COMPUTE=1

# April 2025 optimizations
ENABLE_4BIT_QUANTIZATION=false
ENABLE_FLASH_ATTENTION=false
ENABLE_PROGRESSIVE_LOADING=false
QUANTIZATION_GROUP_SIZE=128
QUANTIZATION_SCHEME="symmetric"
MEMORY_PROFILE=false

# Other settings
BENCHMARK_DB_PATH="./benchmark_db.duckdb"
RESULTS_DIR="./web_platform_results"
CREATE_CHART=true

# Display help
function show_help {
    echo "Web Platform Integration Test Runner (April 2025)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model MODEL           Specific model to test (bert, t5, clip, etc.)"
    echo "  --models MODEL1,MODEL2  Comma-separated list of models to test"
    echo "  --all-models            Test all supported models"
    echo "  --webnn-only            Test only WebNN platform"
    echo "  --webgpu-only           Test only WebGPU platform"
    echo "  --run-webnn             Include WebNN in testing"
    echo "  --run-webgpu            Include WebGPU in testing"
    echo "  --use-browser-automation Use actual browser automation"
    echo "  --browser BROWSER       Browser to use (chrome, edge, firefox)"
    echo ""
    echo "  # March 2025 optimizations"
    echo "  --enable-compute-shaders  Enable compute shader optimization"
    echo "  --enable-shader-precompile Enable shader precompilation optimization"
    echo "  --enable-parallel-loading Enable parallel loading optimization"
    echo "  --firefox                Use Firefox with exceptional compute shader performance (55% improvement)"
    echo "  --firefox-advanced-compute Enable Firefox --MOZ_WEBGPU_ADVANCED_COMPUTE=1 flag for best performance"
    echo "  --disable-compute-shaders Disable compute shader optimization"
    echo "  --disable-shader-precompile Disable shader precompilation optimization"
    echo "  --disable-parallel-loading Disable parallel loading optimization"
    echo "  --march-optimizations   Enable all March 2025 optimizations"
    echo ""
    echo "  # April 2025 optimizations"
    echo "  --enable-4bit-quantization Enable 4-bit quantization for LLMs"
    echo "  --enable-flash-attention Enable Flash Attention for memory efficiency"
    echo "  --enable-progressive-loading Enable progressive tensor loading"
    echo "  --quantization-group-size SIZE Set quantization group size (default: 128)"
    echo "  --quantization-scheme SCHEME Set quantization scheme (symmetric/asymmetric)"
    echo "  --memory-profile        Enable memory profiling"
    echo "  --april-optimizations   Enable all April 2025 optimizations"
    echo ""
    echo "  # General options"
    echo "  --all-optimizations     Enable all optimizations (March + April 2025)"
    echo "  --no-optimizations      Disable all optimizations"
    echo "  --db-path PATH          Path to benchmark database"
    echo "  --results-dir DIR       Directory to store results"
    echo "  --no-charts             Disable chart generation"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model bert --run-webgpu"
    echo "  $0 --models bert,t5,vit --march-optimizations"
    echo "  $0 --all-models --use-browser-automation --browser firefox"
    echo "  $0 --model llama --april-optimizations"
    echo "  $0 --all-optimizations --browser chrome --model whisper"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --all-models)
            MODELS="bert,t5,vit,clip,whisper,wav2vec2,clap,llava,llama,xclip,qwen2,detr"
            shift
            ;;
        --webnn-only)
            WEBNN_ONLY=true
            shift
            ;;
        --webgpu-only)
            WEBGPU_ONLY=true
            shift
            ;;
        --run-webnn)
            RUN_WEBNN=true
            shift
            ;;
        --run-webgpu)
            RUN_WEBGPU=true
            shift
            ;;
        --use-browser-automation)
            USE_BROWSER_AUTOMATION=true
            shift
            ;;
        --browser)
            BROWSER="$2"
            shift 2
            ;;
        --firefox)
            USE_FIREFOX=true
            BROWSER="firefox"
            USE_FIREFOX_ADVANCED_COMPUTE=true
            shift
            ;;
        --firefox-advanced-compute)
            USE_FIREFOX_ADVANCED_COMPUTE=true
            shift
            ;;
        # March 2025 optimizations
        --enable-compute-shaders)
            ENABLE_COMPUTE_SHADERS=true
            shift
            ;;
        --enable-shader-precompile)
            ENABLE_SHADER_PRECOMPILE=true
            shift
            ;;
        --enable-parallel-loading)
            ENABLE_PARALLEL_LOADING=true
            shift
            ;;
        --disable-compute-shaders)
            ENABLE_COMPUTE_SHADERS=false
            shift
            ;;
        --disable-shader-precompile)
            ENABLE_SHADER_PRECOMPILE=false
            shift
            ;;
        --disable-parallel-loading)
            ENABLE_PARALLEL_LOADING=false
            shift
            ;;
        --march-optimizations)
            ENABLE_COMPUTE_SHADERS=true
            ENABLE_SHADER_PRECOMPILE=true
            ENABLE_PARALLEL_LOADING=true
            shift
            ;;
        # April 2025 optimizations
        --enable-4bit-quantization)
            ENABLE_4BIT_QUANTIZATION=true
            shift
            ;;
        --enable-flash-attention)
            ENABLE_FLASH_ATTENTION=true
            shift
            ;;
        --enable-progressive-loading)
            ENABLE_PROGRESSIVE_LOADING=true
            shift
            ;;
        --quantization-group-size)
            QUANTIZATION_GROUP_SIZE="$2"
            shift 2
            ;;
        --quantization-scheme)
            QUANTIZATION_SCHEME="$2"
            shift 2
            ;;
        --memory-profile)
            MEMORY_PROFILE=true
            shift
            ;;
        --april-optimizations)
            ENABLE_4BIT_QUANTIZATION=true
            ENABLE_FLASH_ATTENTION=true
            ENABLE_PROGRESSIVE_LOADING=true
            shift
            ;;
        # General options
        --all-optimizations)
            # Enable both March and April 2025 optimizations
            ENABLE_COMPUTE_SHADERS=true
            ENABLE_SHADER_PRECOMPILE=true
            ENABLE_PARALLEL_LOADING=true
            ENABLE_4BIT_QUANTIZATION=true
            ENABLE_FLASH_ATTENTION=true
            ENABLE_PROGRESSIVE_LOADING=true
            shift
            ;;
        --no-optimizations)
            ENABLE_COMPUTE_SHADERS=false
            ENABLE_SHADER_PRECOMPILE=false
            ENABLE_PARALLEL_LOADING=false
            ENABLE_4BIT_QUANTIZATION=false
            ENABLE_FLASH_ATTENTION=false
            ENABLE_PROGRESSIVE_LOADING=false
            shift
            ;;
        --db-path)
            BENCHMARK_DB_PATH="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --no-charts)
            CREATE_CHART=false
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Set environment variables for March 2025 optimizations
if [ "$ENABLE_COMPUTE_SHADERS" = true ]; then
    export WEBGPU_COMPUTE_SHADERS_ENABLED=1
    echo "Compute shaders enabled"
else
    unset WEBGPU_COMPUTE_SHADERS_ENABLED
    echo "Compute shaders disabled"
fi

if [ "$ENABLE_SHADER_PRECOMPILE" = true ]; then
    export WEBGPU_SHADER_PRECOMPILE_ENABLED=1
    echo "Shader precompilation enabled"
else
    unset WEBGPU_SHADER_PRECOMPILE_ENABLED
    echo "Shader precompilation disabled"
fi

if [ "$ENABLE_PARALLEL_LOADING" = true ]; then
    export WEB_PARALLEL_LOADING_ENABLED=1
    echo "Parallel loading enabled"
else
    unset WEB_PARALLEL_LOADING_ENABLED
    echo "Parallel loading disabled"
fi

# Set environment variables for April 2025 optimizations
if [ "$ENABLE_4BIT_QUANTIZATION" = true ]; then
    export WEBGPU_DEFAULT_TO_4BIT=1
    export WEBGPU_QUANTIZATION_GROUP_SIZE=$QUANTIZATION_GROUP_SIZE
    export WEBGPU_QUANTIZATION_SCHEME=$QUANTIZATION_SCHEME
    echo "4-bit quantization enabled (group size: $QUANTIZATION_GROUP_SIZE, scheme: $QUANTIZATION_SCHEME)"
else
    unset WEBGPU_DEFAULT_TO_4BIT
    echo "4-bit quantization disabled"
fi

if [ "$ENABLE_FLASH_ATTENTION" = true ]; then
    export WEBGPU_FLASH_ATTENTION=1
    echo "Flash Attention enabled"
else
    unset WEBGPU_FLASH_ATTENTION
    echo "Flash Attention disabled"
fi

if [ "$ENABLE_PROGRESSIVE_LOADING" = true ]; then
    export WEBGPU_PROGRESSIVE_LOADING=1
    echo "Progressive tensor loading enabled"
else
    unset WEBGPU_PROGRESSIVE_LOADING
    echo "Progressive tensor loading disabled"
fi

if [ "$MEMORY_PROFILE" = true ]; then
    export WEBGPU_MEMORY_PROFILING=1
    export WEBGPU_PROFILING_OUTPUT_DIR="$RESULTS_DIR/memory_profiles"
    mkdir -p "$WEBGPU_PROFILING_OUTPUT_DIR"
    echo "Memory profiling enabled (output to: $WEBGPU_PROFILING_OUTPUT_DIR)"
fi

# Enable browser automation if requested
if [ "$USE_BROWSER_AUTOMATION" = true ]; then
    export USE_BROWSER_AUTOMATION=1
    
    if [ "$USE_FIREFOX" = true ]; then
        # Firefox-specific flags for WebGPU compute shader optimization
        export BROWSER_PREFERENCE="firefox"
        echo "Using browser automation with Firefox"
        
        if [ "$USE_FIREFOX_ADVANCED_COMPUTE" = true ]; then
            echo "Adding Firefox advanced compute shader flags (--MOZ_WEBGPU_ADVANCED_COMPUTE=1)..."
            export MOZ_WEBGPU_ADVANCED_COMPUTE=1
        fi
    else
        export BROWSER_PREFERENCE="$BROWSER"
        echo "Using browser automation with $BROWSER"
    fi
fi

# Export database path
export BENCHMARK_DB_PATH="$BENCHMARK_DB_PATH"
echo "Using benchmark database: $BENCHMARK_DB_PATH"

# Determine platforms to test
if [ "$WEBNN_ONLY" = true ]; then
    PLATFORMS="webnn"
elif [ "$WEBGPU_ONLY" = true ]; then
    PLATFORMS="webgpu"
elif [ "$RUN_WEBNN" = true ] && [ "$RUN_WEBGPU" = true ]; then
    PLATFORMS="webnn,webgpu"
elif [ "$RUN_WEBNN" = true ]; then
    PLATFORMS="webnn"
elif [ "$RUN_WEBGPU" = true ]; then
    PLATFORMS="webgpu"
else
    # Default to WebGPU only
    PLATFORMS="webgpu"
fi

echo "Testing platforms: $PLATFORMS"

# Generate timestamp for the run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if models parameter is set
if [ -z "$MODELS" ]; then
    # Run comprehensive optimization tests
    echo "Running comprehensive optimization tests for all model types..."
    
    # Run all optimization tests if enabled
    if [ "$ENABLE_4BIT_QUANTIZATION" = true ] || [ "$ENABLE_FLASH_ATTENTION" = true ] || [ "$ENABLE_PROGRESSIVE_LOADING" = true ]; then
        echo "Running April 2025 optimization tests..."
        if [ -f "test/test_webgpu_4bit_inference.py" ]; then
            python test/test_webgpu_4bit_inference.py --all-models --simulation
        fi
    fi
    
    # Run standard optimization tests
    python test/test_web_platform_optimizations.py --all-optimizations
    
    # Also run with the database integration
    echo "Running additional tests with database integration..."
    python test/run_web_platform_tests_with_db.py --db-path "$BENCHMARK_DB_PATH" --all-optimizations
    
    exit 0
fi

# Convert models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Run tests for each model
for MODEL in "${MODEL_ARRAY[@]}"; do
    echo "-----------------------------------------------------"
    echo "Testing model: $MODEL"
    
    # Create model-specific result directory
    MODEL_RESULTS_DIR="$RESULTS_DIR/${MODEL}_${TIMESTAMP}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    # Run April 2025 optimization tests first if enabled (for WebGPU)
    if [[ "$PLATFORMS" == *"webgpu"* ]]; then
        # Run 4-bit quantization tests for LLMs and embedding models
        if [ "$ENABLE_4BIT_QUANTIZATION" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|bert|t5|opt)$ ]]; then
            echo "Running 4-bit quantization tests for $MODEL..."
            if [ -f "test/test_webgpu_4bit_inference.py" ]; then
                python test/test_webgpu_4bit_inference.py --model "$MODEL" --compare-precision --simulation \
                    > "$MODEL_RESULTS_DIR/${MODEL}_4bit_quantization.log" 2>&1 || echo "Warning: 4-bit quantization tests failed for $MODEL"
            fi
        fi
        
        # Run WebGPU optimization tests for model
        echo "Running WebGPU optimization tests for $MODEL..."
        OPTIMIZATION_ARGS=""
        
        # Add March 2025 specific optimizations
        if [ "$ENABLE_COMPUTE_SHADERS" = true ] && [[ "$MODEL" =~ ^(whisper|wav2vec2|clap)$ ]]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --compute-shaders"
            
            # Add Firefox-specific optimizations for audio models
            if [ "$USE_FIREFOX" = true ] && [ "$USE_FIREFOX_ADVANCED_COMPUTE" = true ]; then
                echo "Using Firefox with exceptional WebGPU compute shader performance (55% improvement)..."
                OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --firefox"
            fi
        fi
        
        if [ "$ENABLE_SHADER_PRECOMPILE" = true ]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --shader-precompile"
        fi
        
        if [ "$ENABLE_PARALLEL_LOADING" = true ] && [[ "$MODEL" =~ ^(clip|llava|llava_next|xclip|clap)$ ]]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --parallel-loading"
        fi
        
        # Add April 2025 specific optimizations
        if [ "$ENABLE_FLASH_ATTENTION" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|bert|t5|opt|vit)$ ]]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --flash-attention"
        fi
        
        if [ "$ENABLE_PROGRESSIVE_LOADING" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|llava|llava_next)$ ]]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --progressive-loading"
        fi
        
        if [ "$MEMORY_PROFILE" = true ]; then
            OPTIMIZATION_ARGS="$OPTIMIZATION_ARGS --memory-profile"
        fi
        
        if [ -n "$OPTIMIZATION_ARGS" ]; then
            python test/test_web_platform_optimizations.py --model "$MODEL" $OPTIMIZATION_ARGS \
                > "$MODEL_RESULTS_DIR/${MODEL}_webgpu_optimizations.log" 2>&1 || echo "Warning: WebGPU optimization tests failed for $MODEL"
        fi
    fi
    
    # Run with database integration
    echo "Running database integrated tests for $MODEL..."
    
    for PLATFORM in $(echo $PLATFORMS | tr ',' ' '); do
        echo "Testing platform: $PLATFORM"
        
        # Build command with appropriate flags
        DB_CMD="python test/run_web_platform_tests_with_db.py --model \"$MODEL\" --platform \"$PLATFORM\" --db-path \"$BENCHMARK_DB_PATH\" --output-dir \"$MODEL_RESULTS_DIR\""
        
        # Add appropriate flags for WebGPU
        if [ "$PLATFORM" = "webgpu" ]; then
            # March 2025 optimizations
            if [ "$ENABLE_COMPUTE_SHADERS" = true ] && [[ "$MODEL" =~ ^(whisper|wav2vec2|clap)$ ]]; then
                DB_CMD="$DB_CMD --compute-shaders"
            fi
            
            if [ "$ENABLE_SHADER_PRECOMPILE" = true ]; then
                DB_CMD="$DB_CMD --shader-precompile"
            fi
            
            if [ "$ENABLE_PARALLEL_LOADING" = true ] && [[ "$MODEL" =~ ^(clip|llava|llava_next|xclip|clap)$ ]]; then
                DB_CMD="$DB_CMD --parallel-loading"
            fi
            
            # April 2025 optimizations
            if [ "$ENABLE_4BIT_QUANTIZATION" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|bert|t5|opt)$ ]]; then
                DB_CMD="$DB_CMD --quantization 4bit --quantization-group-size $QUANTIZATION_GROUP_SIZE --quantization-scheme $QUANTIZATION_SCHEME"
            fi
            
            if [ "$ENABLE_FLASH_ATTENTION" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|bert|t5|opt|vit)$ ]]; then
                DB_CMD="$DB_CMD --flash-attention"
            fi
            
            if [ "$ENABLE_PROGRESSIVE_LOADING" = true ] && [[ "$MODEL" =~ ^(llama|qwen2|llava|llava_next)$ ]]; then
                DB_CMD="$DB_CMD --progressive-loading"
            fi
        fi
        
        # Run the command
        eval "$DB_CMD" > "$MODEL_RESULTS_DIR/${MODEL}_${PLATFORM}_db.log" 2>&1 || echo "Warning: Database test failed for $MODEL on $PLATFORM"
        
        # Create chart if requested
        if [ "$CREATE_CHART" = true ]; then
            python test/scripts/benchmark_db_query.py --model "$MODEL" \
                --platform "$PLATFORM" --format chart \
                --output "$MODEL_RESULTS_DIR/${MODEL}_${PLATFORM}_chart.png" \
                --db-path "$BENCHMARK_DB_PATH" \
                > "$MODEL_RESULTS_DIR/${MODEL}_${PLATFORM}_chart.log" 2>&1 || echo "Warning: Chart generation failed for $MODEL on $PLATFORM"
        fi
    done
    
    echo "Tests completed for $MODEL"
done

echo "-----------------------------------------------------"
echo "All web platform tests completed"
echo "Results stored in: $RESULTS_DIR"

# Generate summary report if database queries are available
if [ -f "test/scripts/benchmark_db_query.py" ]; then
    echo "Generating summary report..."
    python test/scripts/benchmark_db_query.py --report web_platform \
        --format html --output "$RESULTS_DIR/web_platform_summary_${TIMESTAMP}.html" \
        --db-path "$BENCHMARK_DB_PATH" \
        > "$RESULTS_DIR/web_platform_summary_${TIMESTAMP}.log" 2>&1 || echo "Warning: Summary report generation failed"
    
    # Generate optimization-specific reports if applicable
    if [ "$ENABLE_4BIT_QUANTIZATION" = true ] || [ "$ENABLE_FLASH_ATTENTION" = true ] || [ "$ENABLE_PROGRESSIVE_LOADING" = true ]; then
        echo "Generating April 2025 optimization reports..."
        python test/scripts/benchmark_db_query.py --report web_optimizations \
            --format html --output "$RESULTS_DIR/web_optimization_report_${TIMESTAMP}.html" \
            --db-path "$BENCHMARK_DB_PATH" \
            > "$RESULTS_DIR/web_optimization_report_${TIMESTAMP}.log" 2>&1 || echo "Warning: Optimization report generation failed"
    fi
    
    if [ "$MEMORY_PROFILE" = true ] && [ -d "$WEBGPU_PROFILING_OUTPUT_DIR" ]; then
        echo "Generating memory profile visualization..."
        if [ -f "test/visualize_memory_usage.py" ]; then
            # Generate model-specific memory visualizations
            for MODEL in "${MODEL_ARRAY[@]}"; do
                python test/visualize_memory_usage.py --model "$MODEL" --platform webgpu \
                    --output html --output-dir "$RESULTS_DIR/memory_analysis_${TIMESTAMP}" \
                    > "$RESULTS_DIR/memory_visualization_${MODEL}_${TIMESTAMP}.log" 2>&1 || echo "Warning: Memory visualization generation failed for $MODEL"
            done
            
            # Generate cross-platform 4-bit analysis if 4-bit quantization is enabled
            if [ "$ENABLE_4BIT_QUANTIZATION" = true ]; then
                echo "Generating cross-platform 4-bit analysis..."
                if [ -f "test/test_cross_platform_4bit.py" ]; then
                    python test/test_cross_platform_4bit.py --model "${MODEL_ARRAY[0]}" --hardware cpu cuda webgpu \
                        --output-report "$RESULTS_DIR/cross_platform_4bit_${TIMESTAMP}.html" \
                        --output-plot "$RESULTS_DIR/cross_platform_4bit_chart_${TIMESTAMP}.png" \
                        > "$RESULTS_DIR/cross_platform_4bit_${TIMESTAMP}.log" 2>&1 || echo "Warning: Cross-platform 4-bit analysis failed"
                fi
            fi
        fi
    fi
fi

echo "All operations completed"