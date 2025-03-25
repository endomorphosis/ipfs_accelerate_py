# Archived GitHub Actions Workflows

This directory contains GitHub Actions workflows that were previously used in the project but have been replaced by a simplified workflow system.

## Archived Workflows

1. **`benchmark_db_ci.yml`**: Created and updated benchmark databases with test results.
2. **`ci_circuit_breaker_benchmark.yml`**: Tested the circuit breaker pattern for fault tolerance.
3. **`hardware_monitoring_integration.yml`**: Tested hardware monitoring functionality.
4. **`integration_tests.yml`**: Ran integration tests across different hardware platforms.
5. **`python-publish.yml`**: Published Python packages to PyPI on release.
6. **`test_and_benchmark.yml`**: Ran tests and benchmarks for the main codebase.
7. **`test_results_integration.yml`**: Collected and integrated test results into the database.
8. **`test_results_ci.yml`**: Processed and reported CI test results.
9. **`update_compatibility_matrix.yml`**: Updated the model compatibility matrix documentation.

## Replacement

These workflows have been replaced by the simplified `model_hardware_tests.yml` workflow, which focuses on:

1. Testing model inference across various hardware backends (CPU, CUDA, ROCm, OpenVINO, MPS, QNN)
2. Measuring performance with basic benchmarks
3. Generating a compatibility matrix report

## Reactivating Archived Workflows

To reactivate any of these workflows:

1. Copy the workflow file from this directory back to the parent directory
2. Remove the `.disabled` extension from the original file

For example:
```bash
# To reactivate the benchmark_db_ci.yml workflow
cp /path/to/repo/.github/workflows/archived/benchmark_db_ci.yml /path/to/repo/.github/workflows/
mv /path/to/repo/.github/workflows/benchmark_db_ci.yml.disabled /path/to/repo/.github/workflows/benchmark_db_ci.yml
```

## Notes

These workflows were archived on March 25, 2025 as part of a GitHub Actions refactoring effort to simplify the CI/CD pipeline.