# Hardware Capability Detector

This module provides comprehensive hardware capability detection for the Distributed Testing Framework, with a focus on detecting and reporting hardware capabilities for optimal task distribution in heterogeneous environments.

## Features

- **Comprehensive Hardware Detection**: Automatically detects various hardware types (CPU, GPU, TPU, NPU, WebGPU, WebNN)
- **Database Integration**: Stores capabilities in DuckDB for persistence and efficient querying
- **Hardware Fingerprinting**: Creates unique identifiers for hardware configurations
- **Browser Detection**: Advanced browser automation for detecting WebGPU and WebNN capabilities
- **Worker Compatibility Search**: Finds workers compatible with specific hardware requirements
- **Task-Hardware Compatibility**: Matches tasks to appropriate hardware based on requirements
- **Performance Profiling**: Basic hardware profiling capabilities for performance estimation
- **Comprehensive DuckDB Integration**: Efficient storage and retrieval of hardware capabilities

## Quick Start

```python
# Create detector
from hardware_capability_detector import HardwareCapabilityDetector

# Initialize with database path (optional)
detector = HardwareCapabilityDetector(
    db_path="./hardware_capabilities.duckdb",
    enable_browser_detection=True  # Enable WebGPU/WebNN detection
)

# Detect all hardware capabilities
capabilities = detector.detect_all_capabilities()

# Print capabilities
print(f"Worker ID: {capabilities.worker_id}")
print(f"OS: {capabilities.os_type} {capabilities.os_version}")
print(f"CPU Count: {capabilities.cpu_count}")
print(f"Total Memory: {capabilities.total_memory_gb:.2f} GB")
print(f"Detected {len(capabilities.hardware_capabilities)} hardware capabilities")

# Store capabilities in database
detector.store_capabilities(capabilities)

# Find workers with specific hardware type
gpu_workers = detector.get_workers_by_hardware_type("gpu")
print(f"Found {len(gpu_workers)} workers with GPU hardware")

# Find compatible workers for specific requirements
compatible_workers = detector.find_compatible_workers(
    hardware_requirements={"hardware_type": "gpu", "vendor": "nvidia"},
    min_memory_gb=8.0,
    preferred_hardware_types=["gpu", "cpu"]
)
print(f"Found {len(compatible_workers)} compatible workers")

# Get capabilities for a specific worker
worker_capabilities = detector.get_worker_capabilities("worker-123")
if worker_capabilities:
    print(f"Worker {worker_capabilities.worker_id} has {len(worker_capabilities.hardware_capabilities)} hardware components")
```

## Command Line Usage

The detector can also be used from the command line:

```bash
# Detect hardware capabilities
python hardware_capability_detector.py --db-path ./hardware_db.duckdb

# Enable browser detection
python hardware_capability_detector.py --enable-browser-detection --db-path ./hardware_db.duckdb

# Specify browser path
python hardware_capability_detector.py --enable-browser-detection --browser-path /path/to/browser --db-path ./hardware_db.duckdb

# Output to JSON file
python hardware_capability_detector.py --output-json hardware_capabilities.json

# Search for workers with GPU hardware
python hardware_capability_detector.py --search-workers gpu --db-path ./hardware_db.duckdb

# Find compatible workers
python hardware_capability_detector.py --find-compatible '{"hardware_type": "gpu", "min_memory_gb": 8.0}'
```

## Integration with Coordinator

The hardware capability detector is designed to integrate with the coordinator component of the Distributed Testing Framework:

```python
# Import required modules
from hardware_capability_detector import HardwareCapabilityDetector
from coordinator_hardware_integration import CoordinatorHardwareIntegration

# Create coordinator instance
coordinator = ...  # Your coordinator instance

# Create hardware integration
hardware_integration = CoordinatorHardwareIntegration(
    coordinator=coordinator,
    db_path="./hardware_db.duckdb",
    enable_browser_detection=True
)

# Initialize integration
await hardware_integration.initialize()
```

With the integration, the coordinator will automatically:

1. Process hardware capabilities during worker registration
2. Store capabilities in the database
3. Use hardware compatibility for task assignment
4. Find optimal workers for tasks based on hardware requirements

## Database Schema

The hardware capability detector creates and uses the following tables in the DuckDB database:

### worker_hardware

Stores basic information about worker nodes:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| worker_id | VARCHAR | Unique worker identifier |
| hostname | VARCHAR | Worker hostname |
| os_type | VARCHAR | Operating system type |
| os_version | VARCHAR | Operating system version |
| cpu_count | INTEGER | Number of CPU cores |
| total_memory_gb | FLOAT | Total system memory in GB |
| fingerprint | VARCHAR | Unique hardware fingerprint |
| last_updated | TIMESTAMP | Last update timestamp |
| metadata | JSON | Additional worker metadata |

### hardware_capabilities

Stores detailed information about hardware components:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| worker_id | VARCHAR | Worker identifier |
| hardware_type | VARCHAR | Hardware type (cpu, gpu, tpu, etc.) |
| vendor | VARCHAR | Hardware vendor (nvidia, amd, intel, etc.) |
| model | VARCHAR | Hardware model name |
| version | VARCHAR | Hardware version |
| driver_version | VARCHAR | Driver version |
| compute_units | INTEGER | Number of compute units |
| cores | INTEGER | Number of cores |
| memory_gb | FLOAT | Hardware memory in GB |
| supported_precisions | JSON | List of supported precision types |
| capabilities | JSON | Detailed hardware capabilities |
| scores | JSON | Performance scores by category |
| last_updated | TIMESTAMP | Last update timestamp |

### hardware_performance

Stores performance metrics for hardware components:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| hardware_capability_id | INTEGER | Foreign key to hardware_capabilities |
| benchmark_type | VARCHAR | Type of benchmark |
| metric_name | VARCHAR | Name of the metric |
| metric_value | FLOAT | Value of the metric |
| units | VARCHAR | Units for the metric |
| run_date | TIMESTAMP | Benchmark run timestamp |
| metadata | JSON | Additional benchmark metadata |

## Configuration Options

The hardware capability detector supports the following configuration options:

- **worker_id**: Optional worker ID (will be auto-generated if not provided)
- **db_path**: Path to DuckDB database for storing results
- **enable_browser_detection**: Whether to enable browser-based detection
- **browser_executable_path**: Path to browser executable for automated detection

## Browser Detection

The hardware capability detector can detect WebGPU and WebNN capabilities using browser automation with Selenium. It supports:

- **Chrome**: Preferred for general WebGPU detection
- **Firefox**: Good for WebGPU with compute shader support (audio models)
- **Edge**: Best for WebNN detection

To enable browser detection:

```python
detector = HardwareCapabilityDetector(
    enable_browser_detection=True,
    browser_executable_path="/path/to/browser"  # Optional
)
```

## Performance Profiling

The detector includes basic hardware profiling capabilities:

```python
results = detector.perform_hardware_profiling(
    worker_id="worker-123",
    hardware_type="gpu",
    benchmark_type="basic"  # "basic", "compute", "memory", or "full"
)
```

## Advanced Feature: Hardware Fingerprinting

The detector creates unique fingerprints for hardware configurations, which can be used to:

- Track hardware changes over time
- Identify identical hardware across different workers
- Validate hardware configuration consistency

```python
fingerprint = detector.generate_hardware_fingerprint(capabilities)
print(f"Hardware fingerprint: {fingerprint}")
```

## Requirements

- Python 3.8+
- DuckDB
- psutil
- Selenium (optional, for browser detection)
- webdriver_manager (optional, for browser detection)

## Testing

Run the test script to validate the hardware capability detector:

```bash
python run_test_hardware_integration.py --db-path ./test_db.duckdb
```