# Hardware Monitoring Test Results

## Test Summary

**Status**: ❌ Failure
**Workflow**: hardware-monitoring-tests
**Run ID**: 1234567890
**Commit**: e6372da76ae43a55d48c98dc5f65e8fcc920e850
**Date**: 2025-03-19 15:47:32

## Test Results

| Metric | Value |
|--------|-------|
| Total Tests | 12 |
| Passed | 10 |
| Failed | 2 |
| Duration | 64.23s |

### Failed Tests

1. `TestHardwareUtilizationMonitor.test_database_integration`
   - **Error**: Failed to connect to database: Connection refused
   - **Location**: `tests/test_hardware_utilization_monitor.py:128`

2. `TestCoordinatorHardwareMonitoringIntegration.test_worker_monitors_created`
   - **Error**: Expected worker monitors to be created, but none were found
   - **Location**: `tests/test_hardware_utilization_monitor.py:285`

## Resource Utilization

| Resource | Average | Peak | Target |
|----------|---------|------|--------|
| CPU | 45.2% | 78.5% | <90% |
| Memory | 1.24 GB | 1.87 GB | <4 GB |
| GPU | 0.0% | 0.0% | N/A |
| Disk | 35.6 MB/s | 124.7 MB/s | <200 MB/s |

## Next Steps

1. Check database connectivity in the test environment
2. Verify worker monitor creation logic
3. Run the test suite locally with `--verbose` to get more detailed output

## Links

- [Test Report](https://github.com/your-org/your-repo/actions/runs/1234567890/artifacts)
- [Test Logs](https://github.com/your-org/your-repo/actions/runs/1234567890)
- [Documentation](https://github.com/your-org/your-repo/blob/main/test/distributed_testing/README_HARDWARE_MONITORING.md)

---

![Hardware Monitoring Tests](https://github.com/your-org/your-repo/raw/main/badges/hardware_monitoring_status.svg)

🤖 *This comment was generated automatically by the Hardware Monitoring CI System*