#!/usr/bin/env python3
"""
Example script demonstrating how to use the Cross-Platform Worker Support module.

This script shows how to detect the current platform, identify hardware capabilities,
create platform-specific deployment scripts, and generate startup scripts for workers.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the CrossPlatformWorkerSupport module
from duckdb_api.distributed_testing.cross_platform_worker_support import CrossPlatformWorkerSupport


def detect_platform_and_hardware():
    """Demonstrate platform detection and hardware identification."""
    # Initialize cross-platform support
    support = CrossPlatformWorkerSupport()
    
    # Get platform information
    platform_info = support.get_platform_info()
    print("\n=== Platform Information ===")
    print(f"Platform: {platform_info['platform']}")
    print(f"System: {platform_info['system']}")
    print(f"Release: {platform_info['release']}")
    print(f"Architecture: {platform_info['architecture']}")
    
    # Get detailed hardware information
    hardware_info = support.detect_hardware()
    print("\n=== Hardware Information ===")
    
    # CPU information
    cpu_info = hardware_info.get('cpu', {})
    print(f"CPU: {cpu_info.get('model', 'Unknown')}")
    print(f"Cores: {cpu_info.get('cores', 'Unknown')}")
    
    # Memory information
    memory_info = hardware_info.get('memory', {})
    print(f"Memory: {memory_info.get('total_gb', 'Unknown')} GB")
    
    # GPU information
    gpu_info = hardware_info.get('gpu', {})
    print(f"GPUs detected: {gpu_info.get('count', 0)}")
    for i, device in enumerate(gpu_info.get('devices', [])):
        print(f"  GPU {i+1}: {device.get('name', 'Unknown')} ({device.get('type', 'generic')})")
        if 'memory' in device:
            print(f"    Memory: {device.get('memory', 'Unknown')}")
    
    # Disk information
    disk_info = hardware_info.get('disk', {})
    print(f"Disk space: {disk_info.get('total_gb', 'Unknown')} GB total, {disk_info.get('free_gb', 'Unknown')} GB free")
    
    # Container-specific information (if applicable)
    if 'container' in hardware_info:
        container_info = hardware_info.get('container', {})
        print("\n=== Container Information ===")
        print(f"Container type: {container_info.get('type', 'Unknown')}")
        if 'id' in container_info:
            print(f"Container ID: {container_info.get('id')}")
        if 'cpu_limit' in container_info:
            print(f"CPU limit: {container_info.get('cpu_limit')} cores")
        if 'memory_limit_gb' in container_info:
            print(f"Memory limit: {container_info.get('memory_limit_gb')} GB")


def create_deployment_scripts(output_dir, coordinator_url, api_key, worker_id):
    """Demonstrate creation of platform-specific deployment scripts."""
    # Initialize cross-platform support
    support = CrossPlatformWorkerSupport()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for the worker
    config = {
        'coordinator_url': coordinator_url,
        'api_key': api_key,
        'worker_id': worker_id
    }
    
    # Create deployment scripts for different platforms
    platform_scripts = {}
    
    # Linux script
    linux_output = os.path.join(output_dir, 'deploy_linux.sh')
    try:
        with mock.patch('platform.system', return_value='Linux'):
            linux_support = CrossPlatformWorkerSupport()
            linux_script = linux_support.create_deployment_script(config, linux_output)
            platform_scripts['linux'] = linux_script
            print(f"Created Linux deployment script: {linux_script}")
    except Exception as e:
        print(f"Error creating Linux script: {e}")
    
    # Windows script
    windows_output = os.path.join(output_dir, 'deploy_windows')
    try:
        with mock.patch('platform.system', return_value='Windows'):
            windows_support = CrossPlatformWorkerSupport()
            windows_script = windows_support.create_deployment_script(config, windows_output)
            platform_scripts['windows'] = windows_script
            print(f"Created Windows deployment script: {windows_script}")
    except Exception as e:
        print(f"Error creating Windows script: {e}")
    
    # macOS script
    macos_output = os.path.join(output_dir, 'deploy_macos.sh')
    try:
        with mock.patch('platform.system', return_value='Darwin'):
            macos_support = CrossPlatformWorkerSupport()
            macos_script = macos_support.create_deployment_script(config, macos_output)
            platform_scripts['macos'] = macos_script
            print(f"Created macOS deployment script: {macos_script}")
    except Exception as e:
        print(f"Error creating macOS script: {e}")
    
    # Container script
    container_output = os.path.join(output_dir, 'docker-compose.yml')
    try:
        with mock.patch('os.path.exists', lambda path: path == '/.dockerenv'):
            container_support = CrossPlatformWorkerSupport()
            container_script = container_support.create_deployment_script(config, container_output)
            platform_scripts['container'] = container_script
            print(f"Created container deployment script: {container_script}")
            print(f"Also created Dockerfile.worker in same directory")
    except Exception as e:
        print(f"Error creating container script: {e}")
    
    return platform_scripts


def generate_startup_scripts(output_dir, coordinator_url, api_key):
    """Demonstrate generation of platform-specific startup scripts."""
    # Initialize cross-platform support for the current platform
    support = CrossPlatformWorkerSupport()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a new worker ID for this example
    import uuid
    worker_id = f"worker_{uuid.uuid4().hex[:8]}"
    
    # Generate startup script for the current platform
    startup_script = support.get_startup_script(
        coordinator_url=coordinator_url,
        api_key=api_key,
        worker_id=worker_id
    )
    
    # Save the startup script to a file
    platform_name = support.current_platform
    script_extension = 'bat' if platform_name == 'windows' else 'sh'
    script_path = os.path.join(output_dir, f"start_worker_{platform_name}.{script_extension}")
    
    with open(script_path, 'w') as f:
        f.write(startup_script)
    
    # Make script executable (not needed for Windows)
    if platform_name != 'windows':
        os.chmod(script_path, 0o755)
    
    print(f"\n=== Generated Startup Script ===")
    print(f"Script saved to: {script_path}")
    print(f"Worker ID: {worker_id}")
    print(f"Platform: {platform_name}")
    
    return script_path


def demonstrate_path_conversion():
    """Demonstrate path conversion for different platforms."""
    # Initialize cross-platform support
    support = CrossPlatformWorkerSupport()
    
    # Example paths to convert
    test_paths = [
        "/home/user/data/file.txt",
        "C:/Users/user/Documents/file.txt",
        "../relative/path/file.txt",
        "~/user/file.txt"
    ]
    
    print("\n=== Path Conversion ===")
    print(f"Current platform: {support.current_platform}")
    
    for path in test_paths:
        converted = support.convert_path_for_platform(path)
        print(f"Original: {path}")
        print(f"Converted: {converted}")
        print("-" * 40)
    
    # Show how paths would be converted on other platforms
    print("\nPath conversion on different platforms:")
    
    # Example path
    example_path = "/data/models/bert/config.json"
    
    print(f"\nOriginal path: {example_path}")
    
    # Linux path
    try:
        with mock.patch('platform.system', return_value='Linux'):
            linux_support = CrossPlatformWorkerSupport()
            linux_path = linux_support.convert_path_for_platform(example_path)
            print(f"Linux: {linux_path}")
    except Exception as e:
        print(f"Error converting for Linux: {e}")
    
    # Windows path
    try:
        with mock.patch('platform.system', return_value='Windows'):
            windows_support = CrossPlatformWorkerSupport()
            windows_path = windows_support.convert_path_for_platform(example_path)
            print(f"Windows: {windows_path}")
    except Exception as e:
        print(f"Error converting for Windows: {e}")
    
    # macOS path
    try:
        with mock.patch('platform.system', return_value='Darwin'):
            macos_support = CrossPlatformWorkerSupport()
            macos_path = macos_support.convert_path_for_platform(example_path)
            print(f"macOS: {macos_path}")
    except Exception as e:
        print(f"Error converting for macOS: {e}")


def main():
    """Main function to run the example script."""
    parser = argparse.ArgumentParser(description="Cross-Platform Worker Support Example")
    parser.add_argument("--detect", action="store_true", help="Detect platform and hardware")
    parser.add_argument("--scripts", action="store_true", help="Create deployment scripts")
    parser.add_argument("--startup", action="store_true", help="Generate startup script")
    parser.add_argument("--paths", action="store_true", help="Demonstrate path conversion")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--coordinator-url", default="http://localhost:8080", help="Coordinator URL")
    parser.add_argument("--api-key", default="example_key_123", help="API key")
    parser.add_argument("--output-dir", default="./output", help="Output directory for scripts")
    
    args = parser.parse_args()
    
    # If no specific actions are requested, show help
    if not (args.detect or args.scripts or args.startup or args.paths or args.all):
        parser.print_help()
        return
    
    # Run requested or all examples
    if args.detect or args.all:
        print("\n" + "="*50)
        print("PLATFORM AND HARDWARE DETECTION EXAMPLE")
        print("="*50)
        detect_platform_and_hardware()
    
    if args.scripts or args.all:
        print("\n" + "="*50)
        print("DEPLOYMENT SCRIPT CREATION EXAMPLE")
        print("="*50)
        create_deployment_scripts(
            args.output_dir, 
            args.coordinator_url, 
            args.api_key, 
            f"worker_example"
        )
    
    if args.startup or args.all:
        print("\n" + "="*50)
        print("STARTUP SCRIPT GENERATION EXAMPLE")
        print("="*50)
        generate_startup_scripts(
            args.output_dir,
            args.coordinator_url,
            args.api_key
        )
    
    if args.paths or args.all:
        print("\n" + "="*50)
        print("PATH CONVERSION EXAMPLE")
        print("="*50)
        demonstrate_path_conversion()


if __name__ == "__main__":
    # Import mock here to avoid dependencies for normal usage
    from unittest import mock
    main()