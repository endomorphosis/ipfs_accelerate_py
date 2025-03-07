#!/bin/bash
# Script to run comprehensive testing of 4-bit inference on WebGPU/WebNN for all 13 high-priority model classes
# This generates a detailed HTML report and compatibility matrix

# Set WEBGPU_SIMULATION and WEBNN_SIMULATION environment variables if not running on real hardware
export WEBGPU_SIMULATION=1
export WEBNN_SIMULATION=1

# Create results directory if it doesn't exist
mkdir -p results/4bit_coverage

echo "Running comprehensive 4-bit inference test for all 13 high-priority model classes..."
echo "Testing on WebGPU and WebNN backends with Chrome, Firefox, and Edge browsers..."

# Run the main test script with all supported hardware and browsers
python test_webgpu_4bit_model_coverage.py \
  --hardware both \
  --browsers chrome firefox edge \
  --output-report results/4bit_coverage/webgpu_4bit_coverage_report.html \
  --output-matrix results/4bit_coverage/webgpu_4bit_compatibility_matrix.html \
  --output-json results/4bit_coverage/webgpu_4bit_coverage_results.json \
  --simulate

echo "Generating additional reports for specific model categories..."

# Test only text models (BERT, T5, LLAMA, Qwen2)
python test_webgpu_4bit_model_coverage.py \
  --models bert t5 llama qwen2 \
  --hardware both \
  --browsers chrome firefox edge \
  --output-report results/4bit_coverage/text_models_4bit_report.html \
  --output-matrix results/4bit_coverage/text_models_4bit_matrix.html \
  --output-json results/4bit_coverage/text_models_4bit_results.json \
  --simulate

# Test only vision models (ViT, DETR)
python test_webgpu_4bit_model_coverage.py \
  --models vit detr \
  --hardware both \
  --browsers chrome firefox edge \
  --output-report results/4bit_coverage/vision_models_4bit_report.html \
  --output-matrix results/4bit_coverage/vision_models_4bit_matrix.html \
  --output-json results/4bit_coverage/vision_models_4bit_results.json \
  --simulate

# Test only audio models (Whisper, Wav2Vec2)
python test_webgpu_4bit_model_coverage.py \
  --models whisper wav2vec2 \
  --hardware both \
  --browsers chrome firefox edge \
  --output-report results/4bit_coverage/audio_models_4bit_report.html \
  --output-matrix results/4bit_coverage/audio_models_4bit_matrix.html \
  --output-json results/4bit_coverage/audio_models_4bit_results.json \
  --simulate

# Test only multimodal models (CLIP, CLAP, LLaVA, LLaVA-Next, XCLIP)
python test_webgpu_4bit_model_coverage.py \
  --models clip clap llava llava_next xclip \
  --hardware both \
  --browsers chrome firefox edge \
  --output-report results/4bit_coverage/multimodal_models_4bit_report.html \
  --output-matrix results/4bit_coverage/multimodal_models_4bit_matrix.html \
  --output-json results/4bit_coverage/multimodal_models_4bit_results.json \
  --simulate

# WebGPU-only test with memory usage tracking enabled
python test_webgpu_4bit_model_coverage.py \
  --hardware webgpu \
  --browsers chrome \
  --output-report results/4bit_coverage/webgpu_memory_usage_report.html \
  --output-json results/4bit_coverage/webgpu_memory_usage_results.json \
  --test-memory-usage \
  --simulate

echo "All tests completed successfully."
echo "Results saved to results/4bit_coverage directory:"
ls -l results/4bit_coverage/

echo "HTML reports and compatibility matrices are available at:"
echo "  - Main report: results/4bit_coverage/webgpu_4bit_coverage_report.html"
echo "  - Compatibility matrix: results/4bit_coverage/webgpu_4bit_compatibility_matrix.html"
echo "  - Category-specific reports are also available in the same directory"

echo "Done!"