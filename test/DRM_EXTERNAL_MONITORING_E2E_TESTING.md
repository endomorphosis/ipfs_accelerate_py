# DRM External Monitoring End-to-End Testing Guide

This guide documents the end-to-end testing implementation for the DRM External Monitoring integration with Prometheus and Grafana. It covers the test architecture, setup, execution, and interpretation of results.

## Overview

The end-to-end test for DRM External Monitoring verifies the complete integration between the Dynamic Resource Management (DRM) system and external monitoring platforms. It validates that:

1. DRM metrics are correctly exported in Prometheus format
2. Prometheus successfully scrapes these metrics
3. Grafana dashboards can be imported and display the metrics
4. Alert configurations work as expected

This test provides comprehensive validation of the entire monitoring pipeline from DRM to visualization and alerting.

## Test Architecture

The end-to-end test implements the following architecture:

![DRM External Monitoring Test Architecture](https://mermaid.ink/img/pako:eNqFU11v2jAU_SuWT5DQLiNQKKB1mqYWVaN0bGvbh0l7MIlLLA1xZJuyo_Df547DDknVlwTn3nPOuR_25RYzSRHM2Ea7uTHHhm3Yo3WnBdKSCgnZYpkzQRF2g9uxhBl85ioS8i_aAFoklXPbwGZcZpVyNlJp2uCMLpljkA8cwXbQ4i67TFEoS4Q5l0Q2LkhYUf0bccErlElVW1gzLVKkOisTVT9JT5j4LETQFzLDhW5Ot6dYU1Jq32JeUpJiKHljSLzMC_18k8QWHKBZjcD-uJsN_8yvJ5fiIa4aZt2GI2iCHuv5J3jtJwRbrtgK4xW2HQxtbBVpqhYLrO0s5pVN2lR0R9OuoOfVfV5xfYdLLSROW4h54i4QP8sSqJT8jmsqN3kCpUE7rnwlFTKCZmfRPp52A3MsUkHIAhAMYy32NLcdFQbYzGZUuOORMw38MNpfXo3DQOdW_OeWWWgFvlqsGZzqAI_KWxXbD2qK5DI8m0vZT65Gk-lkHI6ik-0HpRbj0S6M9nNZ9aPb6eR2FN4xYgdPuifxiGE_yHvR13g_cPRhKn2MrlE8S9O3WY2mKoaXVr0U8p6STNA1pltWbO2EF7c_9fkP5Nw_MrMfhsFT0LsKzwi2vR5-B8PsyGSXnHo3Bk7_AJMb0KQ?)

### Components:

1. **Docker Containers:**
   - Prometheus container for metrics collection
   - Grafana container for dashboard visualization

2. **DRM Components:**
   - Mock DRM implementation for simulated metrics
   - DRM Real-Time Dashboard for metrics collection
   - External Monitoring Bridge for Prometheus integration

3. **Test Verification:**
   - Prometheus API calls to verify metrics presence
   - Grafana API calls to verify dashboard import
   - Alert rule verification

## Prerequisites

To run the end-to-end tests, you need:

1. **Docker** installed and running (for Prometheus and Grafana containers)
2. **Python Dependencies:**
   - prometheus_client
   - requests
   - dash
   - dash_bootstrap_components
   - plotly

3. **Available Ports:**
   - 9191: Prometheus container
   - 9292: Grafana container
   - 9393: DRM metrics exporter
   - 9494: DRM dashboard

## Running the Tests

The test suite can be run using the provided script:

```bash
./run_drm_external_monitoring_e2e_test.sh
```

This script:
1. Checks for required dependencies
2. Verifies port availability
3. Creates test output directory
4. Executes the end-to-end tests
5. Cleans up containers afterward

## Test Phases

The end-to-end test executes in five phases:

1. **Setup Phase:**
   - Create temporary test directory
   - Start Prometheus container with test configuration
   - Start Grafana container
   - Configure Prometheus data source in Grafana
   - Start DRM with external monitoring

2. **Prometheus Metrics Validation:**
   - Verify that DRM metrics are collected in Prometheus
   - Check for core metrics like CPU utilization, memory usage, etc.

3. **Grafana Dashboard Validation:**
   - Import the auto-generated Grafana dashboard
   - Verify dashboard structure and panels

4. **Alert Configuration Testing:**
   - Configure test alert rules for Prometheus
   - Verify alert configuration and evaluation

5. **Tear Down Phase:**
   - Stop all processes
   - Remove Docker containers
   - Clean up temporary files

## Test Results Interpretation

After running the tests, results are stored in the `e2e_test_output` directory. The main output file is `test_output.log`, which contains:

- Setup and teardown logs
- Test execution details
- Pass/fail status for each test phase
- Any errors or failures encountered

A successful test run will show all test phases passing. If any phase fails, the log will indicate which specific verification failed and why.

## Troubleshooting

Common issues and their solutions:

1. **Port Conflicts:**
   - The test uses ports 9191, 9292, 9393, and 9494
   - Ensure these ports are not in use before running tests
   - To check: `sudo lsof -i :<port_number>`

2. **Docker Issues:**
   - Ensure Docker daemon is running: `systemctl status docker`
   - Verify sufficient permissions: `docker ps`
   - If containers don't start, check Docker logs: `docker logs test_prometheus_drm`

3. **Python Dependency Issues:**
   - Install all dependencies: `pip install -r requirements_dashboard.txt`
   - Check Python version compatibility (requires Python 3.7+)

4. **Networking Issues:**
   - Ensure host.docker.internal resolves correctly in Docker containers
   - If using Linux, you may need to use the `--add-host` flag (already done in the script)

5. **Failed Metrics Collection:**
   - Check that the DRM dashboard started correctly
   - Verify Prometheus scraping configuration
   - Check metrics endpoint is accessible: `curl http://localhost:9393/metrics`

## Adding New Tests

To extend the test suite with additional tests:

1. Add new test methods to the `TestDRMExternalMonitoringE2E` class
2. Follow the naming convention `test_XX_description` where XX is the sequence number
3. Create helper functions for complex verification logic
4. Update the documentation to reflect new test cases

## Best Practices

When running or modifying these tests:

1. Run tests on a system with sufficient resources (at least 4GB RAM for Docker containers)
2. Don't run tests on production systems where port conflicts might occur
3. Always check the test output logs for detailed information
4. If tests fail, clean up containers manually: `docker rm -f test_prometheus_drm test_grafana_drm`
5. Update test ports if needed to avoid conflicts with other services

## Related Documentation

- [EXTERNAL_MONITORING_INTEGRATION_GUIDE.md](EXTERNAL_MONITORING_INTEGRATION_GUIDE.md) - Main integration guide
- [REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md](REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md) - DRM dashboard guide