#!/bin/bash

# Run Comprehensive Test Generator for HuggingFace Transformers
# This script provides convenience commands for running the generator in different modes

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GENERATOR_SCRIPT="$SCRIPT_DIR/comprehensive_test_generator.py"
OUTPUT_DIR="$SCRIPT_DIR/../refactored_test_suite"

function print_help {
    echo "HuggingFace Transformers Test Generator"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  discover         Only discover classes, don't generate tests"
    echo "  all              Generate tests for all model classes"
    echo "  vision           Generate tests for vision models"
    echo "  text             Generate tests for encoder-only and decoder-only models"
    echo "  speech           Generate tests for speech/audio models"
    echo "  multimodal       Generate tests for multimodal models"
    echo "  custom           Generate tests for custom set of models (requires --classes or --categories)"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show what would be done without generating files"
    echo "  --overwrite      Overwrite existing test files"
    echo "  --workers N      Use N parallel workers (default: 4)"
    echo "  --output DIR     Use DIR as output directory (default: ../refactored_test_suite)"
    echo "  --classes X Y Z  Generate tests for classes starting with X, Y, Z"
    echo "  --categories X Y Generate tests for categories X, Y"
    echo "  --verbose        Show verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 discover                   # Discover all HuggingFace classes"
    echo "  $0 all --dry-run              # Show what tests would be generated for all classes"
    echo "  $0 vision --overwrite         # Generate tests for vision models, overwriting existing files"
    echo "  $0 custom --classes BERT GPT2 # Generate tests for BERT and GPT2 models"
    echo ""
}

# Function to check if Python and required packages are installed
function check_requirements {
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is required but not installed. Please install Python 3."
        exit 1
    fi
    
    echo "Checking for required Python packages..."
    python3 -c "import transformers" 2>/dev/null || {
        echo "HuggingFace Transformers is required but not installed."
        echo "Please install it with: pip install transformers"
        exit 1
    }
}

# Parse command line arguments
COMMAND=""
DRY_RUN=""
OVERWRITE=""
WORKERS="4"
CLASSES=""
CATEGORIES=""
VERBOSE=""

if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --classes)
            CLASSES="--classes"
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CLASSES="$CLASSES $1"
                shift
            done
            ;;
        --categories)
            CATEGORIES="--categories"
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CATEGORIES="$CATEGORIES $1"
                shift
            done
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Check requirements
check_requirements

# Run the appropriate command
case "$COMMAND" in
    discover)
        echo "Discovering all HuggingFace Transformers classes..."
        python3 "$GENERATOR_SCRIPT" --discover-only --discovery-output transformers_classes.json $VERBOSE
        echo "Discovered classes saved to transformers_classes.json"
        ;;
    all)
        echo "Generating tests for all model classes..."
        python3 "$GENERATOR_SCRIPT" --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    vision)
        echo "Generating tests for vision models..."
        python3 "$GENERATOR_SCRIPT" --categories vision --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    text)
        echo "Generating tests for text models (encoder-only and decoder-only)..."
        python3 "$GENERATOR_SCRIPT" --categories encoder_only decoder_only --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    speech)
        echo "Generating tests for speech/audio models..."
        python3 "$GENERATOR_SCRIPT" --categories speech --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    multimodal)
        echo "Generating tests for multimodal models..."
        python3 "$GENERATOR_SCRIPT" --categories multimodal --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    custom)
        if [ -z "$CLASSES" ] && [ -z "$CATEGORIES" ]; then
            echo "Error: custom command requires --classes or --categories options"
            print_help
            exit 1
        fi
        echo "Generating tests for custom model selection..."
        python3 "$GENERATOR_SCRIPT" $CLASSES $CATEGORIES --output-dir "$OUTPUT_DIR" --max-workers "$WORKERS" $DRY_RUN $OVERWRITE $VERBOSE
        ;;
    help)
        print_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        print_help
        exit 1
        ;;
esac

echo "Done!"