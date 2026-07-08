#!/bin/bash
# Script to archive old GitHub Actions workflows

# Create archive directory if it doesn't exist
mkdir -p /home/barberb/ipfs_accelerate_py/.github/workflows/archived

# List of workflows to archive
WORKFLOWS=(
  "benchmark_db_ci.yml"
  "ci_circuit_breaker_benchmark.yml"
  "hardware_monitoring_integration.yml"
  "integration_tests.yml"
  "python-publish.yml"
  "test_and_benchmark.yml"
  "test_results_integration.yml"
  "update_compatibility_matrix.yml"
  "test_results_ci.yml"
)

# Archive each workflow
for workflow in "${WORKFLOWS[@]}"; do
  if [ -f "/home/barberb/ipfs_accelerate_py/.github/workflows/$workflow" ]; then
    echo "Archiving $workflow"
    cp "/home/barberb/ipfs_accelerate_py/.github/workflows/$workflow" "/home/barberb/ipfs_accelerate_py/.github/workflows/archived/$workflow"
    # Add .disabled extension to original file to disable it
    mv "/home/barberb/ipfs_accelerate_py/.github/workflows/$workflow" "/home/barberb/ipfs_accelerate_py/.github/workflows/$workflow.disabled"
  else
    echo "Workflow $workflow not found"
  fi
done

echo "Archived workflows to /home/barberb/ipfs_accelerate_py/.github/workflows/archived/"
echo "Original workflows have been disabled with .disabled extension"