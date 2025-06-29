# Cross-Platform Worker Support

## Overview

The Cross-Platform Worker Support module provides a unified interface for deploying and managing worker nodes across different operating systems and container environments within the Distributed Testing Framework. This module ensures consistent functionality regardless of the underlying platform while maximizing platform-specific capabilities.

## Features

- **Platform Detection**: Automatically identifies the current operating system or container environment
- **Hardware Detection**: Discovers available CPU, memory, GPU, and disk resources specific to each platform
- **Deployment Script Generation**: Creates platform-specific deployment scripts for worker nodes
- **Startup Script Generation**: Produces scripts to start workers with proper environment configuration
- **Dependency Management**: Handles installation of required dependencies with platform-specific methods
- **Path Conversion**: Ensures correct path formats for different operating systems
- **Container Support**: Specialized support for Docker and Kubernetes environments

## Supported Platforms

- Linux (Ubuntu, CentOS, Debian, etc.)
- Windows
- macOS
- Containers (Docker, Kubernetes)

## Installation

The module is included with the Distributed Testing Framework and requires no separate installation. Required dependencies will be installed automatically when using the `install_dependencies()` method.

## Usage

### Basic Usage

```python
from duckdb_api.distributed_testing.cross_platform_worker_support import CrossPlatformWorkerSupport

# Initialize for the current platform
support = CrossPlatformWorkerSupport()

# Detect hardware capabilities
hardware_info = support.detect_hardware()
print(f"Platform: {hardware_info['platform']}")
print(f"CPU: {hardware_info['cpu']['cores']} cores, {hardware_info['cpu']['model']}")
print(f"Memory: {hardware_info['memory']['total_gb']} GB")
print(f"GPUs: {hardware_info['gpu']['count']}")
```

### Creating Deployment Scripts

```python
# Configuration for worker deployment
config = {
    "coordinator_url": "http://coordinator.example.com:8080",
    "api_key": "your_api_key",
    "worker_id": "worker_123",
    "log_to_file": True
}

# Create a deployment script appropriate for the current platform
output_path = "./deploy_worker"  # Extension will be added based on platform
script_path = support.create_deployment_script(config, output_path)
print(f"Created deployment script at {script_path}")
```

### Generating Startup Scripts

```python
# Generate a worker startup script
startup_script = support.get_startup_script(
    coordinator_url="http://coordinator.example.com:8080",
    api_key="your_api_key",
    worker_id="worker_abc123"
)

# Save to a file
script_extension = "bat" if support.current_platform == "windows" else "sh"
with open(f"start_worker.{script_extension}", "w") as f:
    f.write(startup_script)

# Make executable on Unix-like systems
if support.current_platform != "windows":
    import os
    os.chmod(f"start_worker.{script_extension}", 0o755)
```

### Installing Dependencies

```python
# Install default dependencies for the current platform
support.install_dependencies()

# Install specific dependencies
custom_deps = ["websockets", "psutil", "pyjwt", "numpy", "duckdb"]
support.install_dependencies(dependencies=custom_deps)
```

### Path Conversion

```python
# Convert paths to the correct format for the current platform
linux_path = "/home/user/data/models/bert.onnx"
windows_path = "C:\\Users\\user\\Documents\\models\\bert.onnx"

# These will be converted to the appropriate format for the current platform
path1 = support.convert_path_for_platform(linux_path)
path2 = support.convert_path_for_platform(windows_path)
```

### Getting Platform Information

```python
# Get detailed information about the current platform
platform_info = support.get_platform_info()
print(json.dumps(platform_info, indent=2))
```

## Example Script

The module includes a comprehensive example script that demonstrates all functionality:

```bash
# Run with all examples
python duckdb_api/distributed_testing/examples/cross_platform_worker_example.py --all

# Run specific examples
python duckdb_api/distributed_testing/examples/cross_platform_worker_example.py --detect  # Platform and hardware detection
python duckdb_api/distributed_testing/examples/cross_platform_worker_example.py --scripts  # Deployment script creation
python duckdb_api/distributed_testing/examples/cross_platform_worker_example.py --startup  # Startup script generation
python duckdb_api/distributed_testing/examples/cross_platform_worker_example.py --paths  # Path conversion
```

## Hardware Detection Details

The module detects different hardware components with platform-specific methods:

### CPU Detection

- **Linux**: Uses `/proc/cpuinfo` to get CPU model and core count
- **Windows**: Uses WMI or `wmic` fallback to get CPU information
- **macOS**: Uses `sysctl` to get CPU details and detects Apple Silicon
- **Container**: Detects cgroup limits for containerized environments

### Memory Detection

- **Linux**: Uses `/proc/meminfo` to get total memory
- **Windows**: Uses `psutil` for memory information
- **macOS**: Uses `sysctl hw.memsize` to get total memory
- **Container**: Detects memory limits from cgroups

### GPU Detection

- **NVIDIA GPUs**: Uses `nvidia-smi` to get GPU information (all platforms)
- **AMD GPUs**: Uses `rocm-smi` on Linux/Windows
- **Intel/Integrated GPUs**: Platform-specific detection methods
- **Apple GPUs**: Uses `system_profiler` on macOS for Apple Silicon detection

### Disk Detection

- All platforms use `shutil.disk_usage()` for consistent disk information

## Testing

The module includes a comprehensive test suite that verifies all functionality:

```bash
# Run the test suite
python -m unittest duckdb_api.distributed_testing.test_cross_platform_worker_support
```

## Architecture

The module uses a handler pattern to manage platform-specific implementations:

1. `CrossPlatformWorkerSupport`: Main class that provides the unified interface
2. `PlatformHandler`: Abstract base class for platform-specific handlers
3. Platform-specific implementations:
   - `LinuxPlatformHandler`: Linux-specific implementation
   - `WindowsPlatformHandler`: Windows-specific implementation
   - `MacOSPlatformHandler`: macOS-specific implementation
   - `ContainerPlatformHandler`: Container-specific implementation

This design allows for:
- Easy addition of new platform support
- Clean separation of platform-specific code
- Unified interface regardless of platform

## Extending for Custom Platforms

To add support for a new platform:

1. Create a new handler class that inherits from `PlatformHandler`
2. Implement all required methods with platform-specific code
3. Update the `platform_handlers` dictionary in `CrossPlatformWorkerSupport.__init__`

## Best Practices

- Always use `convert_path_for_platform` when dealing with file paths
- Use the platform-specific deployment scripts for consistent worker setup
- Let the module handle platform detection rather than manually checking
- Use hardware detection results to inform resource allocation

## License

This module is part of the IPFS Accelerate Python Framework and is subject to the same license terms.