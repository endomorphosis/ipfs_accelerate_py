# IPFS Accelerate Python SDK Documentation

**Date:** March 7, 2025  
**Version:** 0.4.0

## Overview

The IPFS Accelerate Python SDK is a comprehensive toolkit for accelerating IPFS operations using hardware acceleration. It provides a unified interface for working with various hardware platforms, optimizing content delivery through P2P networking, and storing benchmark results in a database.

### Key Features

- **Hardware Acceleration**: Automatic detection and utilization of available hardware (CPU, GPU, WebNN, WebGPU)
- **IPFS Integration**: Optimized IPFS content loading and distribution
- **P2P Optimization**: Enhanced content distribution through peer-to-peer network optimization
- **Database Integration**: Built-in storage and analysis of acceleration results
- **Cross-Platform Support**: Works across diverse hardware and browser environments
- **Browser-Specific Optimizations**: Special optimizations for different browsers (e.g., Firefox for audio models)

### Architecture

The SDK consists of these core components:

1. **IPFS Integration Layer**: Interfaces with IPFS for content loading and storage.
2. **P2P Network Optimizer**: Optimizes content distribution across peers.
3. **Hardware Acceleration Layer**: Detects and utilizes available hardware for acceleration.
4. **Database Integration**: Stores and analyzes benchmark results.
5. **Configuration Manager**: Manages SDK settings and preferences.

## Installation

### Requirements

- Python 3.7 or newer
- DuckDB for database integration (optional): `pip install duckdb pandas`
- PyTorch for GPU acceleration (optional): `pip install torch`
- Selenium and Websockets for WebNN/WebGPU (optional): `pip install selenium websockets`

### Installation Process

```bash
# Clone the repository
git clone https://github.com/your-organization/ipfs-accelerate-py.git

# Navigate to the directory
cd ipfs-accelerate-py

# Install requirements
pip install -r requirements.txt
```

## Core Components

### IPFSAccelerate Class

The central class that coordinates all SDK functionality.

```python
from ipfs_accelerate_py import IPFSAccelerate

# Create an instance
accelerator = IPFSAccelerate()

# Load a checkpoint from IPFS
result = accelerator.load_checkpoint_and_dispatch("QmHash...")

# Get a file from IPFS
file_result = accelerator.get_file("QmHash...", output_path="./output.data")

# Add a file to IPFS
add_result = accelerator.add_file("./my_file.data")

# Get P2P network analytics
analytics = accelerator.get_p2p_network_analytics()
```

### Hardware Acceleration

The SDK automatically detects available hardware and selects the optimal platform for acceleration.

```python
from ipfs_accelerate_py import (
    accelerate, detect_hardware, get_optimal_hardware, 
    get_hardware_details, is_real_hardware
)

# Detect available hardware
available_hardware = detect_hardware()
print(f"Available hardware: {available_hardware}")

# Get optimal hardware for a model
optimal_hardware = get_optimal_hardware("bert-base-uncased", model_type="text")
print(f"Optimal hardware for BERT: {optimal_hardware}")

# Get hardware details
cuda_details = get_hardware_details("cuda")
print(f"CUDA details: {cuda_details}")

# Check if real hardware (not simulation)
if is_real_hardware("webgpu"):
    print("Real WebGPU hardware is available")
else:
    print("WebGPU is simulated")

# Accelerate a model using the best available hardware
result = accelerate("bert-base-uncased", "This is a test")
print(f"Acceleration result: {result}")
```

### Database Integration

The SDK includes a database integration layer for storing and analyzing acceleration results.

```python
from ipfs_accelerate_py import (
    DatabaseHandler, store_acceleration_result, 
    get_acceleration_results, generate_report
)

# Create a custom database handler
db = DatabaseHandler(db_path="./my_database.duckdb")

# Store a result
result = accelerate("bert-base-uncased", "This is a test")
db.store_acceleration_result(result)

# Get results
results = db.get_acceleration_results(model_name="bert-base-uncased")
print(f"Found {len(results)} results")

# Generate a report
report = db.generate_report(format="markdown", output_file="report.md")

# Use the global database handler
store_acceleration_result(result)
results = get_acceleration_results(limit=10)
report = generate_report(format="html", output_file="report.html")
```

### Configuration

The SDK's behavior can be customized through the configuration manager.

```python
from ipfs_accelerate_py import config

# Create a configuration instance
cfg = config()

# Get configuration values
debug_mode = cfg.get("general", "debug", False)
cache_enabled = cfg.get("cache", "enabled", True)

# Set configuration values
cfg.set("general", "debug", True)
cfg.set("cache", "max_size_mb", 2000)
```

## Usage Examples

### Basic Acceleration

```python
from ipfs_accelerate_py import accelerate

# Accelerate a text model
text_result = accelerate(
    model_name="bert-base-uncased",
    content="This is a test of IPFS acceleration."
)
print(f"Processing time: {text_result['processing_time']:.3f} seconds")
print(f"Throughput: {text_result['throughput_items_per_sec']:.2f} items/second")
print(f"Using hardware: {text_result['hardware']}")

# Accelerate a vision model
vision_result = accelerate(
    model_name="vit-base",
    content={"image_path": "test_image.jpg"},
    config={"hardware": "cuda"}  # Explicitly specify hardware
)
```

### Advanced Configuration

```python
from ipfs_accelerate_py import accelerate

# Advanced configuration options
result = accelerate(
    model_name="whisper-tiny",
    content={"audio_path": "test_audio.mp3"},
    config={
        "hardware": "webgpu",         # Use WebGPU
        "browser": "firefox",         # Use Firefox
        "precision": 8,               # Use 8-bit precision
        "mixed_precision": True,      # Use mixed precision
        "use_firefox_optimizations": True,  # Use Firefox audio optimizations
        "p2p_optimization": True,     # Use P2P optimization
        "store_results": True,        # Store results in database
        "keep_web_implementation": False  # Close web implementation after inference
    }
)
```

### Cross-Platform Testing

```python
from ipfs_accelerate_py import accelerate, detect_hardware

# Get available hardware
available_hardware = detect_hardware()

# Test on all available hardware
results = {}
for hardware in available_hardware:
    try:
        print(f"Testing on {hardware}...")
        result = accelerate(
            model_name="bert-base-uncased",
            content="This is a cross-platform test.",
            config={"hardware": hardware}
        )
        results[hardware] = {
            "latency_ms": result["latency_ms"],
            "throughput": result["throughput_items_per_sec"],
            "memory_mb": result["memory_usage_mb"]
        }
    except Exception as e:
        print(f"Error on {hardware}: {e}")

# Print results
for hw, metrics in results.items():
    print(f"{hw}: {metrics['latency_ms']:.2f} ms, {metrics['throughput']:.2f} items/s")
```

### Browser-Specific Optimizations

```python
from ipfs_accelerate_py import accelerate

# Test Firefox audio optimizations
firefox_result = accelerate(
    model_name="whisper-tiny",
    content={"audio_path": "test_audio.mp3"},
    config={
        "hardware": "webgpu",
        "browser": "firefox",
        "use_firefox_optimizations": True
    }
)

# Test same model on Chrome
chrome_result = accelerate(
    model_name="whisper-tiny",
    content={"audio_path": "test_audio.mp3"},
    config={
        "hardware": "webgpu",
        "browser": "chrome"
    }
)

# Compare results
firefox_throughput = firefox_result["throughput_items_per_sec"]
chrome_throughput = chrome_result["throughput_items_per_sec"]
improvement = (firefox_throughput / chrome_throughput - 1) * 100

print(f"Firefox throughput: {firefox_throughput:.2f} items/second")
print(f"Chrome throughput: {chrome_throughput:.2f} items/second")
print(f"Firefox improvement: {improvement:.1f}%")
```

### Database Analysis

```python
from ipfs_accelerate_py import accelerate, DatabaseHandler

# Create database handler
db = DatabaseHandler()

# Run tests for multiple hardware platforms
hardware_platforms = ["cpu", "cuda", "webgpu"]
model_name = "bert-base-uncased"
content = "This is a test for database analysis."

for hardware in hardware_platforms:
    # Run acceleration
    result = accelerate(
        model_name=model_name,
        content=content,
        config={"hardware": hardware}
    )
    print(f"Tested {hardware}: {result['latency_ms']:.2f} ms")

# Generate report
report = db.generate_report(format="markdown", output_file="hardware_comparison.md")
print("Report generated: hardware_comparison.md")
```

## API Reference

### IPFSAccelerate Class

```python
class IPFSAccelerate:
    def __init__(self, config_instance=None, backends_instance=None, 
                 p2p_optimizer_instance=None, hardware_acceleration_instance=None, 
                 db_handler_instance=None)
    
    def load_checkpoint_and_dispatch(self, cid: str, endpoint: Optional[str] = None, 
                                    use_p2p: bool = True) -> Dict[str, Any]
    
    def add_file(self, file_path: str) -> Dict[str, Any]
    
    def get_file(self, cid: str, output_path: Optional[str] = None, 
                 use_p2p: bool = True) -> Dict[str, Any]
    
    def get_p2p_network_analytics(self) -> Dict[str, Any]
```

### HardwareDetector Class

```python
class HardwareDetector:
    def __init__(self, config_instance=None)
    
    def detect_hardware(self) -> List[str]
    
    def get_hardware_details(self, hardware_type: str = None) -> Dict[str, Any]
    
    def is_real_hardware(self, hardware_type: str) -> bool
    
    def get_optimal_hardware(self, model_name: str, model_type: str = None) -> str
```

### HardwareAcceleration Class

```python
class HardwareAcceleration:
    def __init__(self, config_instance=None)
    
    def accelerate(self, model_name, content, config=None)
    
    async def accelerate_web(self, model_name, content, platform="webgpu", browser="chrome", 
                            precision=16, mixed_precision=False, firefox_optimizations=False)
    
    def accelerate_torch(self, model_name, content, hardware="cuda")
```

### DatabaseHandler Class

```python
class DatabaseHandler:
    def __init__(self, db_path=None)
    
    def store_acceleration_result(self, result)
    
    def get_acceleration_results(self, model_name=None, hardware_type=None, limit=100)
    
    def generate_report(self, format="markdown", output_file=None)
    
    def close()
```

### P2PNetworkOptimizer Class

```python
class P2PNetworkOptimizer:
    def __init__(self, config_instance=None)
    
    def start()
    
    def stop()
    
    def optimize_retrieval(self, cid, timeout_seconds=5.0)
    
    def optimize_content_placement(self, cid, replica_count=3)
    
    def get_performance_stats()
```

### Utility Functions

```python
# Core functions
def accelerate(model_name: str, content: Any, config: Dict[str, Any] = None) -> Dict[str, Any]
def detect_hardware() -> List[str]
def get_optimal_hardware(model_name: str, model_type: str = None) -> str
def get_hardware_details(hardware_type: str = None) -> Dict[str, Any]
def is_real_hardware(hardware_type: str) -> bool

# Database functions
def store_acceleration_result(result)
def get_acceleration_results(model_name=None, hardware_type=None, limit=100)
def generate_report(format="markdown", output_file=None)

# System information
def get_system_info() -> Dict[str, Any]
```

## Best Practices

1. **Hardware Selection**:
   - Let the SDK automatically select the optimal hardware with `accelerate()` rather than specifying a hardware type
   - For specific testing, explicitly set the hardware with `config={"hardware": "cuda"}`

2. **Browser Optimization**:
   - Use Firefox for audio models to benefit from optimized compute shaders
   - Use Edge for WebNN acceleration
   - Use Chrome for general WebGPU acceleration

3. **Database Usage**:
   - Always store acceleration results for later analysis
   - Use the reporting functionality to track performance across runs
   - Keep database files in version control for historical tracking

4. **P2P Optimization**:
   - Enable P2P optimization for better content distribution
   - Use the `get_p2p_network_analytics()` function to monitor network health

5. **Resource Management**:
   - Close the database connection with `db_handler.close()` when finished
   - Set `keep_web_implementation=False` when done with web acceleration

## Troubleshooting

### Common Issues

1. **Hardware Detection Failures**:
   ```python
   # Check system info for troubleshooting
   import ipfs_accelerate_py as ipfs
   system_info = ipfs.get_system_info()
   print(f"System: {system_info['system']} {system_info['version']}")
   print(f"Available hardware: {system_info['available_hardware']}")
   ```

2. **Browser Automation Issues**:
   ```python
   # Set environment variables before importing
   import os
   os.environ["USE_BROWSER_AUTOMATION"] = "1"
   os.environ["BROWSER_PATH"] = "/path/to/browser"
   
   # Then import and use the SDK
   import ipfs_accelerate_py as ipfs
   ```

3. **Database Connection Errors**:
   ```python
   # Specify an explicit database path
   import ipfs_accelerate_py as ipfs
   db = ipfs.DatabaseHandler(db_path="./my_database.duckdb")
   
   # Check if database is available
   if db.db_available:
       print("Database connection successful")
   else:
       print("Database connection failed")
   ```

4. **P2P Optimization Issues**:
   ```python
   # Check P2P network health
   import ipfs_accelerate_py as ipfs
   analytics = ipfs.get_p2p_network_analytics()
   
   if analytics["status"] == "disabled":
       print("P2P optimization is disabled")
   else:
       print(f"P2P network health: {analytics['network_health']}")
       print(f"Peers: {analytics['peer_count']}")
       print(f"Network efficiency: {analytics['network_efficiency']:.2f}")
   ```

## Advanced Topics

### Custom Hardware Acceleration

You can implement custom hardware acceleration by extending the `HardwareAcceleration` class:

```python
from ipfs_accelerate_py import HardwareAcceleration

class CustomHardwareAcceleration(HardwareAcceleration):
    def __init__(self, config_instance=None):
        super().__init__(config_instance)
    
    def accelerate_custom(self, model_name, content):
        # Custom acceleration logic
        return {
            "status": "success",
            "model_name": model_name,
            "hardware": "custom",
            "processing_time": 0.1,
            # Other metrics...
        }
    
    def accelerate(self, model_name, content, config=None):
        if config and config.get("use_custom", False):
            return self.accelerate_custom(model_name, content)
        return super().accelerate(model_name, content, config)
```

### Integration with Other Frameworks

The SDK can be integrated with other deep learning frameworks:

```python
import tensorflow as tf
from ipfs_accelerate_py import accelerate

# Load a TensorFlow model
model = tf.keras.models.load_model("my_model.h5")

# Define a wrapper function for acceleration
def accelerated_predict(input_data):
    # Use IPFS acceleration
    result = accelerate(
        model_name="my_tensorflow_model",
        content=input_data,
        config={"custom_model": model}
    )
    
    # Extract prediction from result
    return result["prediction"]

# Use the accelerated prediction
prediction = accelerated_predict(my_input_data)
```

### Custom Database Schema

You can extend the database schema for custom metrics:

```python
from ipfs_accelerate_py import DatabaseHandler

class CustomDatabaseHandler(DatabaseHandler):
    def __init__(self, db_path=None):
        super().__init__(db_path)
        self._ensure_custom_schema()
    
    def _ensure_custom_schema(self):
        """Add custom tables to the schema."""
        if not self.connection:
            return
            
        try:
            # Create custom table
            self.connection.execute("""
            CREATE TABLE IF NOT EXISTS custom_metrics (
                id INTEGER PRIMARY KEY,
                acceleration_id INTEGER,
                custom_metric1 FLOAT,
                custom_metric2 FLOAT,
                custom_data JSON,
                FOREIGN KEY (acceleration_id) REFERENCES ipfs_acceleration_results(id)
            )
            """)
            
        except Exception as e:
            logger.error(f"Error ensuring custom schema: {e}")
    
    def store_custom_metrics(self, acceleration_id, metrics):
        """Store custom metrics."""
        if not self.db_available or not self.connection:
            return False
            
        try:
            self.connection.execute("""
            INSERT INTO custom_metrics (
                acceleration_id, custom_metric1, custom_metric2, custom_data
            ) VALUES (?, ?, ?, ?)
            """, [
                acceleration_id,
                metrics.get("custom_metric1", 0),
                metrics.get("custom_metric2", 0),
                json.dumps(metrics)
            ])
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing custom metrics: {e}")
            return False
```

## Release Notes

### Version 0.4.0 (March 7, 2025)

- Added hardware detection and acceleration
- Integrated database storage and reporting
- Enhanced `accelerate()` function with hardware awareness
- Added browser-specific optimizations for audio models
- Improved P2P network optimization
- Updated documentation and examples

### Version 0.3.0 (Previous Release)

- Added WebNN/WebGPU integration
- Basic P2P network optimization
- Initial hardware support

## Further Reading

- [API Documentation](API_DOCUMENTATION.md)
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md)
- [WebNN/WebGPU Integration Guide](WEBNN_WEBGPU_INTEGRATION_GUIDE.md)
- [Database Integration Guide](DATABASE_INTEGRATION_GUIDE.md)
- [P2P Network Optimization Guide](P2P_NETWORK_OPTIMIZATION_GUIDE.md)