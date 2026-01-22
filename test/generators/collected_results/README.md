# Collected Results Directory

This directory stores the actual results from model inference tests collected during test runs. These results are compared against the expected outputs stored in the `expected_results` directory to detect regressions or unexpected behavior.

## Directory Structure

Results are organized by model type, hardware combination, and timestamp:

```
collected_results/
├── bert/
│   ├── cpu/
│   │   ├── 20250311_120000/
│   │   └── ...
│   ├── cuda/
│   │   ├── 20250311_120000/
│   │   └── ...
│   └── ...
├── vit/
│   ├── cpu/
│   │   ├── 20250311_120000/
│   │   └── ...
│   └── ...
└── ...
```

## File Format

Each result file contains:
- Input data used for the test
- Actual output tensor(s)
- Actual performance metrics (latency, throughput)
- Actual memory usage
- Hardware specific details used for the test
- Test timestamp and environment information

## Cleanup Policy

This directory is automatically cleaned by the testing framework, which retains:
- Most recent successful results (to track performance changes)
- Failed test results until they are resolved
- Results needed for reporting and visualization

To manually clean old results:
```bash
python generators/runners/end_to_end/run_e2e_tests.py --clean-old-results --days 14
```

## Usage in CI/CD

In continuous integration environments, results are collected and analyzed automatically. Differences between collected and expected results trigger alerts and block integration of changes until resolved.

Test failures generate detailed reports with visualizations of input, expected output, and actual output to aid in debugging.