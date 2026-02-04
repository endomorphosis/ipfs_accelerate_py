#\!/bin/bash
# Script to generate the remaining critical priority models

SKILLS_DIR=$(dirname $(realpath $0))
OUTPUT_DIR="${SKILLS_DIR}/fixed_tests"
GENERATOR="${SKILLS_DIR}/test_generator_fixed.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "===== Generating Critical Priority Models ====="
echo "Output directory: ${OUTPUT_DIR}"

# Function to generate a model test file
generate_model() {
    local model_name=$1
    local task_type=$2
    
    echo "Generating test for ${model_name}..."
    
    python "${GENERATOR}" --generate "${model_name}" --output-dir "${OUTPUT_DIR}" --task "${task_type}"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully generated ${model_name}"
    else
        echo "❌ Failed to generate ${model_name}"
    fi
}

# Generate tests for the four critical priority models
echo "=== Decoder-only Models ==="
generate_model "gpt_j" "text-generation"

echo "=== Encoder-decoder Models ==="
generate_model "flan_t5" "text2text-generation"

echo "=== Encoder-only Models ==="
generate_model "xlm_roberta" "fill-mask"

echo "=== Vision-text Models ==="
generate_model "vision_text_dual_encoder" "image-classification"

echo "===== Generation Complete ====="

# Verify syntax of the generated files
echo "Verifying syntax of generated files..."
for test_file in "${OUTPUT_DIR}/test_hf_gpt_j.py" "${OUTPUT_DIR}/test_hf_flan_t5.py" "${OUTPUT_DIR}/test_hf_xlm_roberta.py" "${OUTPUT_DIR}/test_hf_vision_text_dual_encoder.py"; do
    if [ -f "${test_file}" ]; then
        python -m py_compile "${test_file}"
        if [ $? -eq 0 ]; then
            echo "✅ ${test_file} - Valid syntax"
        else
            echo "❌ ${test_file} - Syntax error"
        fi
    else
        echo "⚠️ ${test_file} - File not found"
    fi
done
