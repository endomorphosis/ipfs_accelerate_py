#\!/bin/bash
set -e

# Create archive directory if it doesn't exist
mkdir -p /home/barberb/ipfs_accelerate_py/test/archive

# Archive old JSON files
declare -a json_files=(
  "benchmark_results.json"
  "benchmark_results_parallel.json"
  "bert_benchmark_results.json"
  "batch_size_results.json"
  "test_results.json"
  "benchmark_maintenance_report.json"
  "endpoint_fix_results_20250301_130000.json"
  "hardware_detection_results.json"
  "key_models_status_20250302_185214.json"
  "key_models_status_20250302_200055.json"
  "model_dependencies.json"
  "quantization_results.json"
  "quantized_llm_benchmark.json"
  "test_implementation_status.json"
  "unified_framework_status.json"
  "benchmark_progress.json"
  "test_coverage_summary.json"
)

for file in "${json_files[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$file" ] && mv "/home/barberb/ipfs_accelerate_py/test/$file" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Count JSON files moved
echo "JSON files moved to archive directory: $(find /home/barberb/ipfs_accelerate_py/test/archive -name "*.json" | wc -l)"
