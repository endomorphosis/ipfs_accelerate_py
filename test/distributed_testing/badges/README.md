# Status Badges

This directory contains status badges for various components of the Distributed Testing Framework. These badges are automatically updated by CI workflows and show the current status of tests and other metrics.

## Available Badges

### Hardware Monitoring Tests

[![Hardware Monitoring Tests](hardware_monitoring_status.svg)](../README_HARDWARE_MONITORING.md)

This badge shows the current status of the hardware monitoring tests. It is automatically updated by the CI workflow whenever tests are run.

## Badge Format

The badges are available in SVG format for embedding in Markdown files. Each badge also has a corresponding JSON file with the same name that provides the badge data in a machine-readable format.

## Usage

To include a badge in your Markdown files:

```markdown
![Hardware Monitoring Tests](https://github.com/your-org/your-repo/raw/main/test/distributed_testing/badges/hardware_monitoring_status.svg)
```

## Customization

The badge generation is handled by the `generate_status_badge.py` script in the root directory. You can customize the badge style by modifying the script or passing different style parameters to the GitHub workflow.

Available badge styles:
- `flat`: Default style with rounded corners
- `flat-square`: Square corners
- `plastic`: 3D effect
- `for-the-badge`: Larger with all caps
- `social`: Rounded with different font

## Automation

The badges are automatically updated by the hardware monitoring CI workflow. The workflow:

1. Runs the hardware monitoring tests
2. Generates a new badge based on the test results
3. Commits the updated badge to the repository

The badge is updated whether the tests pass or fail, ensuring it always reflects the current status.

## Documentation

For more information about the hardware monitoring system and its CI integration, see:
- [README_HARDWARE_MONITORING.md](../README_HARDWARE_MONITORING.md)
- [README_CI_INTEGRATION.md](../README_CI_INTEGRATION.md)
- [TEST_SUITE_GUIDE.md](../TEST_SUITE_GUIDE.md)