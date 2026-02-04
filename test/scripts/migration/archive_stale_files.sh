#\!/bin/bash
set -e

# Create archive directory if it doesn't exist
mkdir -p /home/barberb/ipfs_accelerate_py/test/archive

# Move backup files
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "*.bak*" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;

# Move old database files
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "benchmark_db*.duckdb.bak*" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "benchmark_db*.duckdb.backup*" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "benchmark_db_*.duckdb" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;
[ -f /home/barberb/ipfs_accelerate_py/test/benchmark_db_old.duckdb ] && mv /home/barberb/ipfs_accelerate_py/test/benchmark_db_old.duckdb /home/barberb/ipfs_accelerate_py/test/archive/

# Move old reports
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "benchmark_report_*.md" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "stale_report_cleanup_report_*.md" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;

# Move duplicated test files with _fix versions
find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "*_fix*.py" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;

# Move one-time utility scripts
declare -a utility_scripts=(
  "apply_syntax_fixes.py"
  "archive_old_md_files.sh"
  "update_hw_guide.sh"
  "update_paths.sh"
  "detect_capabilities_patch.py"
  "apply_simulation_detection_fixes.py"
  "cleanup_stale_reports.py"
  "apply_endpoint_handler_fix.py"
  "fix_api_modules.py"
  "fix_endpoint_handler.py"
  "fix_string_issues.py"
  "fix_generator_simple.py"
  "fix_generators_simple.py"
  "cleanup_db_issues.py"
  "fix_benchmark_db_schema.py"
)

for script in "${utility_scripts[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$script" ] && mv "/home/barberb/ipfs_accelerate_py/test/$script" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Move old implementation versions
declare -a old_implementations=(
  "merged_test_generator_clean.py"
  "integrated_skillset_generator_clean.py"
  "integrated_skillset_generator.py.bak*"
  "fixed_merged_test_generator_clean.py"
  "fixed_merged_test_generator_backup.py"
  "fixed_template_generator.py"
  "minimal_generator.py"
  "benchmark_progress.json"
  "benchmark_results.json"
  "benchmark_results_parallel.json"
  "bert_benchmark_results.json"
  "run_benchmark_with_simulation_detection.py"
  "run_fixed_test_generator.py"
  "run_model_benchmarks.py.bak*"
  "implementation_generator.py.bak*"
  "bert_test_with_qualcomm_fixed.py"
  "test_groq_fix.py"
  "test_handler_fix.py"
  "test_qnn_simulation_fix.py"
)

for impl in "${old_implementations[@]}"; do
  find /home/barberb/ipfs_accelerate_py/test -maxdepth 1 -name "$impl" -type f -exec mv {} /home/barberb/ipfs_accelerate_py/test/archive/ \;
done

# Move old test runners
declare -a old_runners=(
  "run_actual_benchmarks.sh"
  "run_benchmark_timing_report.py"
  "run_cleanup_stale_reports.py"
  "run_fixed_test_generator.py"
  "run_generators_phase16.sh"
  "run_improved_generators.sh"
  "run_key_model_fixes.sh"
  "run_local_benchmark_with_ci.sh"
  "run_openvino_benchmarks.sh"
  "run_phase16_generators.sh"
  "run_queue_backoff_tests.py"
  "run_template_system_check.py"
  "run_test_generator_against_all_skills.py"
  "run_web_quantization_tests.sh"
  "run_webgpu_4bit_model_coverage.sh"
  "run_webgpu_4bit_tests.sh"
  "run_webnn_benchmark.sh"
  "run_webnn_coverage_tests.py"
  "run_webnn_quantized_tests.sh"
  "run_webnn_webgpu_precision_test.sh"
  "run_webnn_webgpu_quantization.sh"
  "sample_webnn_webgpu_test.sh"
)

for runner in "${old_runners[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$runner" ] && mv "/home/barberb/ipfs_accelerate_py/test/$runner" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Count files moved
echo "Files moved to archive directory: $(find /home/barberb/ipfs_accelerate_py/test/archive -type f | wc -l)"
