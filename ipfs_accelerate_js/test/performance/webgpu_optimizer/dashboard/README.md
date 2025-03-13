# WebGPU Optimizer Performance Dashboard

This directory contains the performance dashboard for the WebGPU optimizer benchmarks.

## Overview

The WebGPU Optimizer Performance Dashboard provides an interactive visualization of benchmark results from the WebGPU optimization tests. It helps analyze the performance impact of different optimization techniques across various browsers, operation types, and neural network patterns.

## Features

- **Interactive Charts**: Visualize performance metrics with interactive charts
- **Data Tables**: View detailed benchmark results in tabular format
- **Filtering**: Filter results by operation type, browser, optimization type, etc.
- **Historical Tracking**: Monitor performance changes over time
- **Browser Comparison**: Compare performance across different browsers
- **Multiple Visualization Types**: Bar charts, line charts, radar charts, and more

## Dashboard Sections

1. **Summary Tab**: Overview of all optimization types with key metrics
2. **Operation Fusion Tab**: Detailed results for operation fusion benchmarks
3. **Memory Layout Tab**: Performance impact of memory layout optimizations
4. **Browser-Specific Tab**: Browser-specific optimization results
5. **Neural Network Patterns Tab**: Results from neural network pattern recognition
6. **Browser Comparison Tab**: Performance comparison across different browsers
7. **History Tab**: Performance trends over time

## Generating the Dashboard

### Using npm scripts

```bash
npm run benchmark:webgpu:dashboard
```

### Using the shell script

```bash
./test/run_webgpu_benchmarks.sh --dashboard-only
```

### Using Node.js directly

```bash
node test/performance/webgpu_optimizer/dashboard/generate_dashboard.js
```

## Customizing the Dashboard

You can customize the dashboard generation through command-line options:

```bash
node test/performance/webgpu_optimizer/dashboard/generate_dashboard.js \
  --results-dir=./my_benchmark_results \
  --output-dir=./my_dashboard \
  --output-file=dashboard.html \
  --history-file=./benchmark_history.json \
  --max-history=100 \
  --no-open \
  --verbose
```

## Dashboard Directory Structure

- **template.html**: HTML template for the dashboard
- **generate_dashboard.js**: Script to generate the dashboard
- **assets/**: Directory containing CSS styles and other assets
- **README.md**: This file

## Requirements

- Node.js (v14 or higher)
- Web browser with JavaScript support

## Contributing

To contribute to the dashboard:

1. Modify the dashboard template in `template.html`
2. Add any necessary styles in `assets/dashboard.css`
3. Update the dashboard generation script in `generate_dashboard.js`
4. Test your changes by generating the dashboard with sample benchmark results

## Troubleshooting

If you encounter issues with the dashboard:

- Ensure benchmark results exist in the results directory
- Check for valid JSON format in the benchmark result files
- Verify that the template file contains valid HTML
- Check browser console for JavaScript errors

For more information, see the [WebGPU Optimizer Testing Guide](../OPTIMIZER_TESTING_GUIDE.md).