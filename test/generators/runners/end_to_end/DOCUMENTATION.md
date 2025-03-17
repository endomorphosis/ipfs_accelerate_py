# End-to-End Testing Framework Documentation

## Architecture Overview

The End-to-End Testing Framework provides a comprehensive solution for testing AI models across different hardware platforms. The framework follows a modular architecture with several key components:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Component Generator │───►│  Test Execution     │───►│ Result Validation   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Hardware Detection  │    │ Performance Metrics │    │ Documentation Gen   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          └──────────────────┬──────────────────┬──────────────┘
                             ▼                  ▼
                    ┌─────────────────┐ ┌─────────────────┐    ┌─────────────────┐
                    │ File Storage    │ │ DuckDB Storage  │───►│ Visualization   │
                    └─────────────────┘ └─────────────────┘    │ Dashboard       │
                                                                └─────────────────┘
```

### Core Components

1. **E2ETester Class**: The main class that orchestrates the entire testing process
2. **ResultComparer**: Handles advanced validation of test results against expected outputs
3. **ModelDocGenerator**: Generates comprehensive documentation for model implementations
4. **HardwareDetection**: Detects available hardware platforms and tracks simulation status
5. **DatabaseIntegration**: Stores test results with rich metadata in DuckDB
6. **VisualizationDashboard**: Interactive web dashboard for visualizing test results and performance metrics

## Detailed Component Description

### E2ETester

The `E2ETester` class is the main entry point for the testing framework. It handles:

- Model and hardware selection
- Component generation
- Test execution
- Result validation
- Documentation generation
- Database integration
- Distributed execution

Key methods:
- `run_tests()`: Main entry point for test execution
- `_generate_components()`: Creates skill, test, and benchmark files
- `_run_test()`: Executes tests and collects results
- `_compare_with_expected()`: Compares results using the ResultComparer
- `_store_results()`: Stores results in files and database
- `_generate_documentation()`: Creates documentation using ModelDocGenerator

### ResultComparer

The `ResultComparer` class provides advanced comparison capabilities for test results:

- Configurable tolerance for numeric comparisons
- Specialized tensor comparison with element-wise validation
- Statistical comparison for large arrays
- Detailed difference reporting

Features:
- Support for various data types (numeric, string, list, dict, tensor)
- Recursive deep comparison of nested structures
- Tolerance-based comparison for floating-point values
- Statistical metrics for large tensor outputs

### ModelDocGenerator

The `ModelDocGenerator` creates comprehensive Markdown documentation for model implementations:

- Extracts implementation details from source files
- Formats code snippets for readability
- Includes expected results and examples
- Documents hardware-specific optimizations

Generated documentation includes:
- Model overview and architecture
- Skill implementation details
- Test implementation
- Benchmark implementation
- Expected results
- Hardware-specific notes
- Performance characteristics

### Hardware Detection

The hardware detection system determines which platforms are available:

- Detects CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, WebGPU, and Samsung NPU
- Provides detailed device information
- Tracks simulation status for unavailable hardware
- Supports force-simulation mode for testing

Implementation details:
- Uses try/except blocks to safely detect hardware
- Leverages appropriate libraries for each platform
- Provides detailed device names and capabilities
- Tracks simulation status for transparent reporting

### Database Integration

The database integration stores test results in DuckDB:

- Stores comprehensive test metadata
- Captures environment and platform information
- Tracks git commit details
- Records CI/CD environment variables
- Supports transaction-based storage

Stored metadata includes:
- Model and hardware details
- Test status and timestamp
- Performance metrics
- Simulation status
- Platform information
- Git repository details
- CI/CD environment

## Workflow In Action

### 1. Initialization

```python
# Parse command-line arguments
parser = argparse.ArgumentParser(description="End-to-End Testing Framework for IPFS Accelerate")
# ... add arguments ...
args = parser.parse_args()

# Initialize the E2ETester
tester = E2ETester(args)
```

### 2. Test Execution

```python
# Run the tests
results = tester.run_tests()

# Generate a summary report
tester.generate_summary_report(results)
```

### 3. Result Processing

```python
# For each model and hardware platform:
for model in models:
    for hardware in hardware_platforms:
        # Generate components
        skill_path, test_path, benchmark_path = tester._generate_components(model, hardware)
        
        # Run the test
        result = tester._run_test(model, hardware, skill_path, test_path, benchmark_path)
        
        # Compare with expected results
        comparison = tester._compare_with_expected(model, hardware, result)
        
        # Store results
        tester._store_results(model, hardware, result, comparison)
        
        # Generate documentation
        if args.generate_docs:
            tester._generate_documentation(model, hardware, skill_path, test_path, benchmark_path)
```

## Database Schema

The framework uses DuckDB with the following schema:

```sql
CREATE TABLE test_results (
    id INTEGER PRIMARY KEY,
    test_id VARCHAR,
    model_name VARCHAR,
    hardware_type VARCHAR,
    device_name VARCHAR,
    test_type VARCHAR,
    test_date VARCHAR,
    success BOOLEAN,
    is_simulation BOOLEAN,
    error_message VARCHAR,
    metrics JSON,
    platform_info JSON,
    git_info JSON,
    ci_environment JSON,
    result_data JSON,
    comparison_data JSON,
    details JSON
);
```

### Visualization Dashboard and Integrated Reporting System

The VisualizationDashboard class provides an interactive web dashboard for visualizing test results and performance metrics. It uses Dash and Plotly to create dynamic visualizations that update in real-time.

Key features:
- Real-time monitoring of test execution and results
- Comprehensive performance visualization for model-hardware combinations
- Comparative analysis tools for cross-hardware performance
- Simulation validation visualization 
- Historical trend analysis with statistical significance testing
- Customizable views and filtering options

The dashboard is organized into five main tabs:
1. **Overview**: High-level summary of test results and success rates
2. **Performance Analysis**: Detailed performance metrics by model and hardware
3. **Hardware Comparison**: Comparative analysis of different hardware platforms
4. **Time Series Analysis**: Performance trends over time with statistical analysis
5. **Simulation Validation**: Validation of simulation accuracy against real hardware

#### Basic Dashboard Usage

```bash
# Start the visualization dashboard server
python visualization_dashboard.py

# Start with custom configuration
python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb

# Run in development mode with hot reloading 
python visualization_dashboard.py --debug
```

To install the required dependencies:
```bash
pip install -r dashboard_requirements.txt
```

#### Integrated Visualization and Reports System

The framework now includes an integrated system that combines the Visualization Dashboard with the Enhanced CI/CD Reports Generator, providing a unified interface for both systems:

```bash
# Start the dashboard only
python integrated_visualization_reports.py --dashboard

# Generate reports only
python integrated_visualization_reports.py --reports

# Start dashboard and generate reports
python integrated_visualization_reports.py --dashboard --reports

# Specify database path and automatically open browser
python integrated_visualization_reports.py --dashboard --db-path ./benchmark_db.duckdb --open-browser

# Generate specific report types
python integrated_visualization_reports.py --reports --simulation-validation

# Export dashboard visualizations for offline viewing
python integrated_visualization_reports.py --dashboard-export
```

The integrated system provides:
- Unified command-line interface for dashboard and reports
- Consistent database access across all components
- Report generation based on live dashboard data
- Easy-to-use commands for common scenarios
- Support for both interactive exploration and CI/CD integration

For detailed documentation on the visualization dashboard and integrated system, see:
- [VISUALIZATION_DASHBOARD_README.md](./VISUALIZATION_DASHBOARD_README.md) - Comprehensive dashboard guide
- [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - Detailed system architecture
- [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md) - Solutions for common issues
- [INTEGRATION_SUMMARY.md](./INTEGRATION_SUMMARY.md) - Overview of integration architecture

## Hardware Detection Implementation

The framework includes specialized detection for various hardware platforms:

### OpenVINO Detection

```python
def detect_openvino():
    """Detect if OpenVINO is available and usable."""
    try:
        import openvino
        from openvino.runtime import Core
        core = Core()
        available_devices = core.available_devices
        return len(available_devices) > 0
    except (ImportError, ModuleNotFoundError, Exception):
        return False
```

### Qualcomm QNN Detection

```python
def detect_qnn():
    """Detect if Qualcomm Neural Network SDK is available."""
    try:
        import qnn
        from qnn.messaging import QnnMessageListener
        listener = QnnMessageListener()
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False
```

### WebNN/WebGPU Detection

```python
def detect_web_capabilities(capability="webgpu"):
    """Detect if browser with WebNN or WebGPU capabilities can be launched."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        
        driver = webdriver.Chrome(options=options)
        
        if capability == "webgpu":
            is_supported = driver.execute_script("""
                return 'gpu' in navigator && 'requestAdapter' in navigator.gpu;
            """)
        elif capability == "webnn":
            is_supported = driver.execute_script("""
                return 'ml' in navigator && 'getNeuralNetworkContext' in navigator.ml;
            """)
        else:
            is_supported = False
            
        driver.quit()
        return is_supported
    except Exception:
        return False
```

## Result Comparison Logic

The framework uses a sophisticated result comparison system:

```python
# Initialize ResultComparer with appropriate tolerance settings
comparer = ResultComparer(
    tolerance=0.1,  # 10% general tolerance
    tensor_rtol=1e-5,  # Relative tolerance for tensors
    tensor_atol=1e-7,  # Absolute tolerance for tensors
    tensor_comparison_mode='auto'  # Automatically select comparison mode
)

# Use file-based comparison
comparison_result = comparer.compare_with_file(expected_path, result)
```

## Best Practices for Using the Framework

1. **Use Version Control for Expected Results**
   - Expected results should be checked into version control
   - Update expected results when model behavior changes intentionally

2. **Database Integration**
   - Store results in DuckDB for long-term analysis
   - Query the database for performance trends over time
   - Use SQL to generate custom reports

3. **Documentation Generation**
   - Generate documentation for all production models
   - Include hardware-specific optimizations in docs
   - Document expected performance characteristics

4. **CI/CD Integration**
   - Run tests automatically on pull requests
   - Use simulation mode for CI environment
   - Generate reports for pull request reviews

5. **Troubleshooting**
   - Use `--verbose` to see detailed logs
   - Check simulation status for hardware platforms
   - Adjust tolerance for float comparison issues
   - Examine database error logs when available
   - For dashboard and integrated reports system issues:
     - Refer to the [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) for a comprehensive overview of the system design and component interactions
     - Consult the [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md) for detailed solutions to common issues, including:
       - Dashboard process management issues
       - Database connectivity problems
       - Report generation failures
       - Visualization rendering problems
       - Browser integration challenges
       - CI/CD integration troubleshooting

## Distributed Testing

The framework supports distributed testing with worker threads:

```python
def _run_tests_distributed(self):
    """Run tests using multiple worker threads."""
    import concurrent.futures
    
    tasks = []
    for model in self.models_to_test:
        for hardware in self.hardware_to_test:
            tasks.append((model, hardware))
    
    # Create a thread pool and submit tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as executor:
        # Map tasks to workers and collect results
        future_to_task = {executor.submit(self._process_task, model, hardware): (model, hardware) 
                         for model, hardware in tasks}
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_task):
            model, hardware = future_to_task[future]
            try:
                result = future.result()
                results[(model, hardware)] = result
            except Exception as e:
                logger.error(f"Task for {model} on {hardware} failed: {str(e)}")
                results[(model, hardware)] = {"success": False, "error": str(e)}
    
    return results
```

> **Note**: While a more advanced Distributed Testing Framework is in development (currently 40% complete), its integration with the End-to-End Testing Framework is currently out of scope. The current implementation using worker threads provides sufficient parallelization for most testing needs.

## API Reference

### E2ETester Class

#### Constructor
```python
E2ETester(args)
```

#### Methods
```python
run_tests() -> Dict[Tuple[str, str], Dict[str, Any]]
generate_summary_report(results: Dict[Tuple[str, str], Dict[str, Any]])
_generate_components(model: str, hardware: str) -> Tuple[str, str, str]
_run_test(model: str, hardware: str, skill_path: str, test_path: str, benchmark_path: str) -> Dict[str, Any]
_compare_with_expected(model: str, hardware: str, result: Dict[str, Any]) -> Dict[str, Any]
_store_results(model: str, hardware: str, result: Dict[str, Any], comparison: Dict[str, Any]) -> str
_generate_documentation(model: str, hardware: str, skill_path: str, test_path: str, benchmark_path: str) -> str
_update_expected_results(model: str, hardware: str, result: Dict[str, Any]) -> None
_get_hardware_device_name(hardware: str) -> str
_get_db_connection() -> contextlib.AbstractContextManager
_run_tests_distributed() -> Dict[Tuple[str, str], Dict[str, Any]]
_process_task(model: str, hardware: str) -> Dict[str, Any]
```

### ResultComparer Class

#### Constructor
```python
ResultComparer(tolerance=0.1, tensor_rtol=1e-5, tensor_atol=1e-7, tensor_comparison_mode='auto')
```

#### Methods
```python
compare(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]
compare_with_file(expected_path: str, actual: Dict[str, Any]) -> Dict[str, Any]
statistical_tensor_compare(expected_data: numpy.ndarray, actual_data: numpy.ndarray) -> Dict[str, Any]
deep_compare_tensors(expected: numpy.ndarray, actual: numpy.ndarray, rtol: float, atol: float) -> Dict[str, Any]
```

### ModelDocGenerator Class

#### Constructor
```python
ModelDocGenerator(model_name: str, hardware: str, skill_path: str, test_path: str, benchmark_path: str, expected_results_path: Optional[str] = None, output_dir: Optional[str] = None)
```

#### Methods
```python
generate() -> str
_extract_docstrings(file_path: str) -> str
_extract_code_snippets(file_path: str, file_type: str) -> str
_format_expected_results() -> str
_generate_markdown() -> str
```

## Conclusion

The enhanced End-to-End Testing Framework provides a robust solution for testing AI models across different hardware platforms. With its modular architecture, advanced result validation, comprehensive documentation generation, and database integration, the framework ensures that models work correctly across the entire pipeline.