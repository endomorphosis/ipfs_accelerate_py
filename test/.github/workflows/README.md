# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating tests, benchmarks, and deployments.

## Workflows

### End-to-End Tests (e2e_testing.yml)

This workflow runs comprehensive end-to-end tests for models on various hardware platforms.

Key features:
- Matrix jobs for parallel testing of model families
- Database integration for result storage
- Automated report generation and deployment
- Support for manual triggering with parameters

#### Required Permissions

For the end-to-end testing workflow to function properly, you need to enable the following repository permissions:

1. Go to **Settings > Actions > General > Workflow permissions**
2. Select **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**

For GitHub Pages deployment:
1. Go to **Settings > Pages**
2. Set **Source** to **GitHub Actions**

#### Secrets and Configurations

Required secrets for the workflow:
- None required for standard execution

Optional secrets for additional features:
- `DATABASE_SERVER_URL`: If using a remote database server
- `API_KEY`: If using authenticated API endpoints

### Test and Benchmark (test_and_benchmark.yml)

This workflow runs standard tests and benchmarks.

### Integration Tests (integration_tests.yml)

This workflow runs integration tests across multiple environments.

### Benchmark Database CI (benchmark_db_ci.yml)

This workflow handles benchmark data collection and database updates.

### Other Workflows

- `update_compatibility_matrix.yml`: Updates the hardware compatibility matrix
- `test_results_integration.yml`: Integrates test results into reports

## Adding New Workflows

When adding new workflows, please follow these guidelines:

1. Use consistent naming: `purpose_action.yml`
2. Add triggers for `push`, `pull_request`, and `workflow_dispatch`
3. Set appropriate permissions
4. Include artifact uploads for results
5. Add documentation to this README

## Troubleshooting

Common issues:
- Permission errors: Check the permissions section above
- Missing artifacts: Ensure upload/download artifact steps are correct
- Database connectivity: Verify environment variables for database connections