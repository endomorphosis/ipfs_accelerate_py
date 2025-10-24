# Archived GitHub Actions Workflows

This directory contains GitHub Actions workflows that were previously used in the project but have been replaced by a simplified workflow system or archived due to redundancy.

## Archived Workflows

### Historical Archives (Original Refactoring - March 2025)

1. **`benchmark_db_ci.yml`**: Created and updated benchmark databases with test results.
2. **`ci_circuit_breaker_benchmark.yml`**: Tested the circuit breaker pattern for fault tolerance.
3. **`hardware_monitoring_integration.yml`**: Tested hardware monitoring functionality.
4. **`integration_tests.yml`**: Ran integration tests across different hardware platforms.
5. **`python-publish.yml`**: Published Python packages to PyPI on release.
6. **`test_and_benchmark.yml`**: Ran tests and benchmarks for the main codebase.
7. **`update_compatibility_matrix.yml`**: Updated the model compatibility matrix documentation.

### Recent Archives (Redundancy Cleanup - October 2025)

8. **`ci-arm64.yml`**: **DUPLICATE** of `arm64-ci.yml` - Archived as redundant. The `arm64-ci.yml` workflow provides the same functionality with more features.

9. **`enhanced-ci-cd.yml`**: **OVERLAPPING** with `multiarch-ci.yml` - Archived due to redundancy. This workflow required self-hosted runners and overlapped significantly with the more comprehensive `multiarch-ci.yml` that uses GitHub-hosted runners. The multiarch workflow provides better coverage without requiring self-hosted infrastructure.

10. **`test-runner.yml`**: **SIMPLE TEST** - Archived as non-essential. This was a basic ARM64 runner validation workflow that was not critical for the CI/CD pipeline.

## Current Active Workflows

The following workflows remain active and provide comprehensive CI/CD coverage:

### GitHub-Hosted Runners (Always Available)
- **`amd64-ci.yml`**: Comprehensive AMD64 testing across Python 3.9-3.12
- **`multiarch-ci.yml`**: Multi-architecture testing with QEMU emulation for AMD64/ARM64

### Self-Hosted Runners (Conditional on Availability)
- **`arm64-ci.yml`**: Native ARM64 testing (requires self-hosted ARM64 runner)
- **`package-test.yml`**: Package installation validation (requires self-hosted runner)

## Reasons for Archival

### Redundancy
- `ci-arm64.yml` duplicated `arm64-ci.yml`
- `enhanced-ci-cd.yml` overlapped with `multiarch-ci.yml`

### Self-Hosted Runner Dependency
- Several archived workflows required self-hosted runners that may not be consistently available
- GitHub-hosted runners with QEMU emulation provide adequate multi-architecture testing

### Simplification
- Consolidating workflows reduces maintenance burden
- Clearer separation between GitHub-hosted (always run) and self-hosted (optional) workflows

## Reactivating Archived Workflows

To reactivate any of these workflows:

1. Copy the workflow file from this directory back to the parent directory
2. If the workflow has a `.disabled` extension, remove it

For example:
```bash
# To reactivate the ci-arm64.yml workflow
cp .github/workflows/archived/ci-arm64.yml .github/workflows/ci-arm64.yml
```

⚠️ **Note**: Before reactivating redundant workflows (ci-arm64.yml, enhanced-ci-cd.yml, test-runner.yml), consider whether they provide value over the active workflows. These were archived to reduce redundancy and maintenance complexity.

## Archive History

- **March 25, 2025**: Initial archival of deprecated workflows as part of GitHub Actions refactoring
- **October 24, 2025**: Archived redundant workflows (ci-arm64.yml, enhanced-ci-cd.yml, test-runner.yml) to eliminate duplication and simplify CI/CD pipeline