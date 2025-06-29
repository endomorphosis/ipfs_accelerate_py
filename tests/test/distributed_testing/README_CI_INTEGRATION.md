# CI Integration for Hardware Monitoring

This document provides a quick guide to using the continuous integration features of the hardware monitoring system.

## Overview

The hardware monitoring system includes comprehensive CI integration that automates testing, reporting, and notification. These features ensure that the system is continuously tested and validated as the codebase evolves.

## CI Components

1. **GitHub Actions Workflows**:
   - `hardware_monitoring_tests.yml`: Local workflow for testing the hardware monitoring system
   - `hardware_monitoring_integration.yml`: Global workflow for integrating with the broader project

2. **Notification System**:
   - `ci_notification.py`: Script for sending notifications when tests fail
   - `notification_config.json`: Configuration file for notification channels

3. **Status Badge Generator**:
   - `generate_status_badge.py`: Script for generating status badges
   - Automatic badge updates in the repository

4. **CI Simulation Script**:
   - `run_hardware_monitoring_ci_tests.sh`: Script for local CI testing

## Usage

### Running CI Tests Locally

Use the `run_hardware_monitoring_ci_tests.sh` script to simulate the CI environment locally:

```bash
# Basic usage
./run_hardware_monitoring_ci_tests.sh

# Run full tests
./run_hardware_monitoring_ci_tests.sh --mode full

# Generate status badge
./run_hardware_monitoring_ci_tests.sh --mode full --generate-badge

# Send test notifications
./run_hardware_monitoring_ci_tests.sh --mode full --send-notifications

# Run full CI simulation
./run_hardware_monitoring_ci_tests.sh --mode full --ci-integration --generate-badge --send-notifications
```

### Configuring Notifications

Edit `notification_config.json` to configure notification channels:

```json
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.example.com",
    "to_addresses": ["team@example.com"]
  },
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
    "channel": "#ci-alerts"
  },
  "github": {
    "enabled": true,
    "commit_status": true,
    "pr_comment": true
  }
}
```

### Generating Status Badges

Generate status badges manually:

```bash
# Generate status badge with default settings
python generate_status_badge.py

# Specify database path and output
python generate_status_badge.py --db-path ./my_metrics.duckdb --output-path ./my_badge.svg

# Use a different badge style
python generate_status_badge.py --style flat-square
```

Available badge styles:
- `flat`: Default style with rounded corners
- `flat-square`: Square corners
- `plastic`: 3D effect
- `for-the-badge`: Larger with all caps
- `social`: Rounded with different font

### Example Script

The `examples/ci_integration_example.py` script demonstrates how to use the CI integration features programmatically:

```bash
# Run basic example with standard test mode
python examples/ci_integration_example.py

# Run with badge generation
python examples/ci_integration_example.py --generate-badge

# Run with notifications
python examples/ci_integration_example.py --notification

# Run with full features
python examples/ci_integration_example.py --test-mode full --generate-badge --notification --ci-integration

# Specify custom output directory and database path
python examples/ci_integration_example.py --output-dir ./my_ci_output --db-path ./my_metrics.duckdb
```

The example script shows how to:
- Run hardware monitoring tests with CI integration
- Generate status badges programmatically
- Send notifications based on test results
- Customize output paths and test modes

### Using Status Badges

Embed the status badge in your README files:

```markdown
![Hardware Monitoring Tests](path/to/hardware_monitoring_status.svg)
```

### GitHub Actions Integration

The GitHub Actions workflows automatically run when changes are made to relevant files. You can also manually trigger the workflows:

1. Go to the "Actions" tab in your GitHub repository
2. Select "Hardware Monitoring Tests" or "Hardware Monitoring Integration"
3. Click "Run workflow"
4. Choose the test mode and other options
5. Click "Run workflow" to start the CI process

## Files

- **Workflow files**:
  - `.github/workflows/hardware_monitoring_tests.yml`: Local workflow
  - `.github/workflows/hardware_monitoring_integration.yml`: Global workflow

- **Notification system**:
  - `ci_notification.py`: Notification script
  - `notification_config.json`: Notification configuration

- **Status badge generator**:
  - `generate_status_badge.py`: Badge generator script

- **CI simulation**:
  - `run_hardware_monitoring_ci_tests.sh`: Local CI simulation script

## Further Reading

For more detailed information about the CI integration, see:

- [CI_INTEGRATION_SUMMARY.md](CI_INTEGRATION_SUMMARY.md): Comprehensive documentation
- [TEST_SUITE_GUIDE.md](TEST_SUITE_GUIDE.md): Test suite documentation with CI information