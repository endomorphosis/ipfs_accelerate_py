#\!/bin/bash
set -e

# Create archive directory if it doesn't exist
mkdir -p /home/barberb/ipfs_accelerate_py/test/archive

# Archive completed status reports
declare -a completed_reports=(
  "PHASE16_COMPLETION_REPORT.md"
  "PHASE16_COMPLETION_ANNOUNCEMENT.md"
  "IPFS_ACCELERATION_IMPLEMENTATION_COMPLETE.md"
  "DOCUMENTATION_UPDATE_COMPLETED.md"
  "DATABASE_INTEGRATION_COMPLETE.md"
  "EXECUTION_COMPLETE_REPORT.md"
  "TEST_IPFS_ACCELERATE_DB_INTEGRATION_COMPLETED.md"
  "STALE_REPORTS_CLEANUP_COMPLETED.md"
  "WEB_PLATFORM_SUPPORT_COMPLETED.md"
  "PHASE16_VERIFICATION_REPORT.md"
  "BENCHMARK_COMPLETION_REPORT.md"
  "BENCHMARK_DB_INTEGRATION_SUCCESS.md"
  "DUCKDB_INTEGRATION_COMPLETION_SUMMARY.md"
  "PHASE16_IMPROVEMENTS.md"
  "FIXED_WEB_PLATFORM_OPTIMIZATIONS.md"
  "GENERATOR_IMPROVEMENTS_COMPLETE.md"
  "GENERATOR_IMPROVEMENTS_SUMMARY.md"
)

for report in "${completed_reports[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$report" ] && mv "/home/barberb/ipfs_accelerate_py/test/$report" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Archive superseded implementation documents
declare -a implementation_docs=(
  "WEB_PLATFORM_IMPLEMENTATION_PLAN.md"
  "WEB_PLATFORM_INTEGRATION_PLAN.md"
  "WEB_PLATFORM_INTEGRATION_GUIDE_UPDATED.md"
  "PHASE16_IMPLEMENTATION_SUMMARY.md"
  "PHASE16_HARDWARE_IMPLEMENTATION.md"
  "IMPLEMENTATION_PLAN.md"
  "WEB_PLATFORM_ACTION_PLAN.md"
  "WEB_PLATFORM_IMPLEMENTATION_PRIORITIES.md"
  "WEB_PLATFORM_PRIORITIES.md"
  "NEXT_STEPS_IMPLEMENTATION.md"
  "WEB_PLATFORM_IMPLEMENTATION_NEXT_STEPS.md"
  "PHASE16_IMPLEMENTATION_UPDATE.md"
)

for doc in "${implementation_docs[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$doc" ] && mv "/home/barberb/ipfs_accelerate_py/test/$doc" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Archive older summaries and archived documentation
declare -a archive_docs=(
  "WEBNN_WEBGPU_ARCHIVED_DOCS.md"
  "PHASE16_ARCHIVED_DOCS.md"
  "DOCUMENTATION_CLEANUP_SUMMARY.md"
  "PHASE16_CLEANUP_SUMMARY.md"
  "SUMMARY_OF_WORK.md"
  "COMPREHENSIVE_BENCHMARK_SUMMARY.md"
  "DOCUMENTATION_UPDATE_NOTE.md"
  "SUMMARY_OF_IMPROVEMENTS.md"
  "IPFS_ACCELERATE_SUMMARY.md"
  "PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md"
  "WEB_PLATFORM_INTEGRATION_SUMMARY.md"
  "WEB_PLATFORM_SUMMARY.md"
  "OPENVINO_IMPLEMENTATION_SUMMARY.md"
  "QUALCOMM_IMPLEMENTATION_SUMMARY.md"
  "MODEL_REGISTRY_INTEGRATION.md"
  "TEMPLATE_INTEGRATION_SUMMARY.md"
  "HARDWARE_IMPLEMENTATION_SUMMARY.md"
  "HARDWARE_INTEGRATION_SUMMARY.md"
  "QNN_IMPLEMENTATION_SUMMARY.md"
  "QNN_FIX_SUMMARY.md"
)

for doc in "${archive_docs[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$doc" ] && mv "/home/barberb/ipfs_accelerate_py/test/$doc" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Archive benchmark reports
declare -a benchmark_reports=(
  "webgpu_benchmark_summary.md"
  "IPFS_ACCELERATION_BENCHMARK_REPORT.md"
  "webgpu_benchmark_final_report.md"
  "stale_report_cleanup_report.md"
  "benchmark_summary.md"
  "benchmark_report.md"
  "bert_hardware_comparison.md"
  "final_benchmark_report.md"
  "final_hardware_coverage_report.md"
  "comprehensive_benchmark_analysis.log"
  "TEST_COVERAGE_SUMMARY.md"
  "verification_report.md"
  "hardware_availability_report.md"
  "hardware_coverage_report.md"
  "hardware_report.md"
  "comprehensive_webgpu_benchmark_report.md"
  "enhanced_hardware_coverage_report.md"
)

for report in "${benchmark_reports[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$report" ] && mv "/home/barberb/ipfs_accelerate_py/test/$report" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Archive migration and reorganization documentation
declare -a migration_docs=(
  "MARCH_2025_DB_INTEGRATION_UPDATE.md"
  "MIGRATION_REPORT.md"
  "DATA_MIGRATION_README.md"
  "DATABASE_MIGRATION_STATUS.md"
  "BENCHMARK_REALITY_CHECK.md"
  "BENCHMARK_DB_FIX.md"
  "DATA_MIGRATION_COMPLETION_SUMMARY.md"
)

for doc in "${migration_docs[@]}"; do
  [ -f "/home/barberb/ipfs_accelerate_py/test/$doc" ] && mv "/home/barberb/ipfs_accelerate_py/test/$doc" /home/barberb/ipfs_accelerate_py/test/archive/
done

# Count markdown files moved
echo "Markdown files moved to archive directory: $(find /home/barberb/ipfs_accelerate_py/test/archive -name "*.md" | wc -l)"
